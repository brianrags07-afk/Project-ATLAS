"""
Full-season scoring-state timeline builder for Project ATLAS.

The builder:

- loads the frozen game-outcome artifact once
- loads the canonical season event store once
- builds event-level score transitions for every verified game
- checkpoints completed games
- audits score continuity and final-score agreement
- creates no predictions, identities, concepts, or explanations
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION
from .scoring_state_timeline import (
    SCORING_TIMELINE_VERSION,
    _leader,
    _lead_size,
    _score_change_events,
    _canonical_scoring_attribution,
)


SEASON_BUILDER_VERSION = "1.0.0"

REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT = REPO_ROOT / "data"

GAME_OUTCOME_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "outcomes"
    / "{season}"
    / "game_outcomes.parquet"
)

EVENT_STORE_TEMPLATE = (
    DATA_ROOT
    / "history"
    / "game_cards"
    / "events"
    / "game_events_{season}_regular.parquet"
)

OUTPUT_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "scoring_timelines"
    / "{season}"
)

CHECKPOINT_FILENAME = (
    "scoring_state_timelines_partial.parquet"
)

FAILURE_CHECKPOINT_FILENAME = (
    "scoring_state_timeline_failures_partial.parquet"
)

FINAL_TIMELINE_FILENAME = (
    "scoring_state_timelines.parquet"
)

FINAL_AUDIT_FILENAME = (
    "scoring_state_timeline_audit.parquet"
)

FINAL_FAILURE_FILENAME = (
    "scoring_state_timeline_failures.parquet"
)

METADATA_FILENAME = (
    "scoring_state_timeline_metadata.json"
)


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        destination,
    )


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=str,
        )

        temporary_name = handle.name

    os.replace(
        temporary_name,
        destination,
    )


def _normalize_outcomes(
    outcomes: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "reconstruction_verified",
        "score_sources_verified",
    }

    missing = required - set(
        outcomes.columns
    )

    if missing:
        raise KeyError(
            "Frozen game outcomes missing columns: "
            f"{sorted(missing)}"
        )

    normalized = outcomes.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["game_date"] = pd.to_datetime(
        normalized["game_date"],
        errors="raise",
    ).dt.normalize()

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    normalized["home_score"] = pd.to_numeric(
        normalized["home_score"],
        errors="raise",
    ).astype("int64")

    normalized["away_score"] = pd.to_numeric(
        normalized["away_score"],
        errors="raise",
    ).astype("int64")

    normalized = normalized[
        normalized["atlas_season"].eq(
            int(season)
        )
    ].copy()

    if normalized["game_pk"].duplicated().any():
        raise AssertionError(
            "Frozen outcomes contain duplicate game IDs."
        )

    if not normalized[
        "reconstruction_verified"
    ].fillna(False).all():
        raise AssertionError(
            "Unverified reconstructed games are present."
        )

    if not normalized[
        "score_sources_verified"
    ].fillna(False).all():
        raise AssertionError(
            "Unverified score sources are present."
        )

    return normalized.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _normalize_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
    }

    missing = required - set(
        events.columns
    )

    if missing:
        raise KeyError(
            "Event store missing columns: "
            f"{sorted(missing)}"
        )

    normalized = events.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    numeric_columns = [
        "inning",
        "outs_when_up",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
        "delta_run_exp",
        "delta_home_win_exp",
    ]

    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce",
            )

    normalized["_source_order"] = np.arange(
        len(normalized),
        dtype="int64",
    )

    return normalized.sort_values(
        [
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "_source_order",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _build_one_game(
    outcome: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    game_pk = int(
        outcome["game_pk"]
    )

    season = int(
        outcome["atlas_season"]
    )

    if events.empty:
        raise ValueError(
            f"No event rows found for game {game_pk}."
        )

    scoring = _score_change_events(
        events
    )

    if scoring.empty:
        raise AssertionError(
            f"No scoring state transitions found "
            f"for game {game_pk}."
        )

    home_team = str(
        outcome["home_team"]
    )

    away_team = str(
        outcome["away_team"]
    )

    records: list[
        dict[str, Any]
    ] = []

    previous_non_tie_leader: (
        str | None
    ) = None

    for event_number, row in enumerate(
        scoring.itertuples(
            index=False,
        ),
        start=1,
    ):
        pre_home_score = int(
            row.canonical_pre_home_score
        )

        pre_away_score = int(
            row.canonical_pre_away_score
        )

        post_home_score = int(
            row.post_home_score
        )

        post_away_score = int(
            row.post_away_score
        )

        home_runs = int(
            row.home_runs_on_play
        )

        away_runs = int(
            row.away_runs_on_play
        )

        runs_on_play = int(
            row.runs_on_play
        )

        source_inning = int(
            row.inning
        )

        source_inning_half = str(
            row.inning_topbot
        )

        scoring_side = (
            "HOME"
            if home_runs > 0
            else "AWAY"
        )

        attribution = (
            _canonical_scoring_attribution(
                source_inning=
                    source_inning,

                source_inning_half=
                    source_inning_half,

                pitch_number=
                    int(
                        row.pitch_number
                    ),

                scoring_side=
                    scoring_side,

                score_state_repaired=
                    bool(
                        row.score_state_repaired
                    ),
            )
        )

        inning = int(
            attribution["inning"]
        )

        inning_half = str(
            attribution[
                "inning_half"
            ]
        )

        batting_side = str(
            attribution[
                "batting_side"
            ]
        )

        batting_team = (
            away_team
            if batting_side == "AWAY"
            else home_team
        )

        fielding_team = (
            home_team
            if batting_side == "AWAY"
            else away_team
        )

        scoring_team = (
            home_team
            if scoring_side == "HOME"
            else away_team
        )

        pre_leader = _leader(
            pre_home_score,
            pre_away_score,
        )

        post_leader = _leader(
            post_home_score,
            post_away_score,
        )

        prior_non_tie_leader = (
            previous_non_tie_leader
        )

        non_tie_leader_changed = bool(
            post_leader in {
                "HOME",
                "AWAY",
            }
            and prior_non_tie_leader
            in {
                "HOME",
                "AWAY",
            }
            and post_leader
            != prior_non_tie_leader
        )

        if post_leader in {
            "HOME",
            "AWAY",
        }:
            previous_non_tie_leader = (
                post_leader
            )

        play_description = getattr(
            row,
            "des",
            None,
        )

        if (
            play_description is None
            or pd.isna(
                play_description
            )
        ):
            play_description = getattr(
                row,
                "description",
                None,
            )

        records.append({
            "game_pk":
                game_pk,

            "game_date":
                pd.to_datetime(
                    outcome["game_date"]
                ).normalize(),

            "atlas_season":
                season,

            "home_team":
                home_team,

            "away_team":
                away_team,

            "scoring_event_number":
                int(event_number),

            "at_bat_number":
                int(
                    row.at_bat_number
                ),

            "event_pitch_number":
                int(
                    row.pitch_number
                ),

            "source_inning":
                source_inning,

            "source_inning_half":
                source_inning_half,

            "source_batting_side":
                str(
                    attribution[
                        "source_batting_side"
                    ]
                ),

            "inning":
                inning,

            "inning_half":
                inning_half,

            "delayed_score_update":
                bool(
                    attribution[
                        "delayed_score_update"
                    ]
                ),

            "scoring_attribution_repaired":
                bool(
                    attribution[
                        "scoring_attribution_repaired"
                    ]
                ),

            "outs_before_play": (
                None
                if not hasattr(
                    row,
                    "outs_when_up",
                )
                or pd.isna(
                    row.outs_when_up
                )
                else int(
                    row.outs_when_up
                )
            ),

            "batting_side":
                batting_side,

            "batting_team":
                batting_team,

            "fielding_team":
                fielding_team,

            "scoring_side":
                scoring_side,

            "scoring_team":
                scoring_team,

            "batter_id": (
                None
                if not hasattr(
                    row,
                    "batter",
                )
                or pd.isna(
                    row.batter
                )
                else int(
                    row.batter
                )
            ),

            "pitcher_id": (
                None
                if not hasattr(
                    row,
                    "pitcher",
                )
                or pd.isna(
                    row.pitcher
                )
                else int(
                    row.pitcher
                )
            ),

            "pitcher_name":
                getattr(
                    row,
                    "player_name",
                    None,
                ),

            "event_result":
                getattr(
                    row,
                    "events",
                    None,
                ),

            "pitch_description":
                getattr(
                    row,
                    "description",
                    None,
                ),

            "play_description":
                play_description,

            "raw_pre_home_score":
                int(
                    row.raw_pre_home_score
                ),

            "raw_pre_away_score":
                int(
                    row.raw_pre_away_score
                ),

            "pre_home_score":
                pre_home_score,

            "pre_away_score":
                pre_away_score,

            "raw_pre_score_matches_canonical":
                bool(
                    row.raw_pre_score_matches_canonical
                ),

            "score_state_repaired":
                bool(
                    row.score_state_repaired
                ),

            "score_change_within_plate_appearance":
                bool(
                    row.score_change_within_plate_appearance
                ),

            "home_runs_on_play":
                home_runs,

            "away_runs_on_play":
                away_runs,

            "runs_on_play":
                runs_on_play,

            "post_home_score":
                post_home_score,

            "post_away_score":
                post_away_score,

            "post_total_runs":
                int(
                    post_home_score
                    + post_away_score
                ),

            "pre_leader":
                pre_leader,

            "post_leader":
                post_leader,

            "pre_lead_size":
                _lead_size(
                    pre_home_score,
                    pre_away_score,
                ),

            "post_lead_size":
                _lead_size(
                    post_home_score,
                    post_away_score,
                ),

            "tie_created":
                bool(
                    pre_leader != "TIE"
                    and post_leader == "TIE"
                ),

            "tie_broken":
                bool(
                    pre_leader == "TIE"
                    and post_leader != "TIE"
                ),

            "direct_lead_change":
                bool(
                    pre_leader in {
                        "HOME",
                        "AWAY",
                    }
                    and post_leader in {
                        "HOME",
                        "AWAY",
                    }
                    and pre_leader
                    != post_leader
                ),

            "previous_non_tie_leader":
                prior_non_tie_leader,

            "non_tie_leader_changed":
                non_tie_leader_changed,

            "home_leading_after_event":
                post_leader == "HOME",

            "away_leading_after_event":
                post_leader == "AWAY",

            "tied_after_event":
                post_leader == "TIE",

            "delta_run_expectancy": (
                None
                if not hasattr(
                    row,
                    "delta_run_exp",
                )
                or pd.isna(
                    row.delta_run_exp
                )
                else float(
                    row.delta_run_exp
                )
            ),

            "delta_home_win_expectancy": (
                None
                if not hasattr(
                    row,
                    "delta_home_win_exp",
                )
                or pd.isna(
                    row.delta_home_win_exp
                )
                else float(
                    row.delta_home_win_exp
                )
            ),

            "terminal_scoring_event":
                False,

            "final_home_score":
                int(
                    outcome["home_score"]
                ),

            "final_away_score":
                int(
                    outcome["away_score"]
                ),

            "score_sources_verified":
                True,

            "reconstruction_verified":
                True,

            "prediction_created":
                False,

            "identity_updated":
                False,

            "explanation_created":
                False,

            "future_games_used":
                False,

            "brain_engine_version":
                BRAIN_ENGINE_VERSION,

            "scoring_timeline_version":
                SCORING_TIMELINE_VERSION,

            "season_builder_version":
                SEASON_BUILDER_VERSION,
        })

    timeline = pd.DataFrame(
        records
    )

    timeline.loc[
        timeline.index[-1],
        "terminal_scoring_event",
    ] = True

    final_row = timeline.iloc[-1]

    if (
        int(
            final_row[
                "post_home_score"
            ]
        )
        != int(
            outcome["home_score"]
        )
        or int(
            final_row[
                "post_away_score"
            ]
        )
        != int(
            outcome["away_score"]
        )
    ):
        raise AssertionError(
            f"Final scoring state mismatch "
            f"for game {game_pk}."
        )

    if len(timeline) > 1:
        home_continuity = bool(
            (
                timeline[
                    "pre_home_score"
                ].iloc[1:].to_numpy()
                == timeline[
                    "post_home_score"
                ].iloc[:-1].to_numpy()
            ).all()
        )

        away_continuity = bool(
            (
                timeline[
                    "pre_away_score"
                ].iloc[1:].to_numpy()
                == timeline[
                    "post_away_score"
                ].iloc[:-1].to_numpy()
            ).all()
        )

    else:
        home_continuity = True
        away_continuity = True

    if not (
        home_continuity
        and away_continuity
    ):
        raise AssertionError(
            f"Score-state continuity failure "
            f"for game {game_pk}."
        )

    if not timeline[
        "runs_on_play"
    ].gt(0).all():
        raise AssertionError(
            f"Nonpositive scoring transition "
            f"for game {game_pk}."
        )

    if not (
        timeline[
            "home_runs_on_play"
        ].gt(0)
        ^ timeline[
            "away_runs_on_play"
        ].gt(0)
    ).all():
        raise AssertionError(
            f"Invalid scoring-side transition "
            f"for game {game_pk}."
        )

    return timeline.reset_index(
        drop=True
    )


def _audit_one_game(
    timeline: pd.DataFrame,
    outcome: pd.Series,
) -> dict[str, Any]:
    game_pk = int(
        outcome["game_pk"]
    )

    if timeline.empty:
        return {
            "game_pk":
                game_pk,

            "game_date":
                outcome["game_date"],

            "timeline_rows":
                0,

            "timeline_not_empty":
                False,

            "event_numbers_sequential":
                False,

            "score_continuity_home":
                False,

            "score_continuity_away":
                False,

            "positive_score_changes":
                False,

            "one_side_scores_per_event":
                False,

            "one_terminal_event":
                False,

            "terminal_event_is_last":
                False,

            "final_home_score_matches":
                False,

            "final_away_score_matches":
                False,

            "final_score_matches":
                False,

            "unique_event_keys":
                False,

            "provenance_pass":
                False,

            "audit_pass":
                False,
        }

    if len(timeline) > 1:
        home_continuity = bool(
            (
                timeline[
                    "pre_home_score"
                ].iloc[1:].to_numpy()
                == timeline[
                    "post_home_score"
                ].iloc[:-1].to_numpy()
            ).all()
        )

        away_continuity = bool(
            (
                timeline[
                    "pre_away_score"
                ].iloc[1:].to_numpy()
                == timeline[
                    "post_away_score"
                ].iloc[:-1].to_numpy()
            ).all()
        )

    else:
        home_continuity = True
        away_continuity = True

    event_numbers_sequential = bool(
        timeline[
            "scoring_event_number"
        ].tolist()
        == list(
            range(
                1,
                len(timeline) + 1,
            )
        )
    )

    positive_score_changes = bool(
        timeline[
            "runs_on_play"
        ].gt(0).all()
    )

    one_side_scores = bool(
        (
            timeline[
                "home_runs_on_play"
            ].gt(0)
            ^ timeline[
                "away_runs_on_play"
            ].gt(0)
        ).all()
    )

    one_terminal_event = bool(
        int(
            timeline[
                "terminal_scoring_event"
            ].sum()
        ) == 1
    )

    terminal_event_is_last = bool(
        timeline[
            "terminal_scoring_event"
        ].iloc[-1]
    )

    final_home_matches = bool(
        int(
            timeline[
                "post_home_score"
            ].iloc[-1]
        )
        == int(
            outcome["home_score"]
        )
    )

    final_away_matches = bool(
        int(
            timeline[
                "post_away_score"
            ].iloc[-1]
        )
        == int(
            outcome["away_score"]
        )
    )

    event_keys_unique = bool(
        not timeline.duplicated(
            subset=[
                "game_pk",
                "scoring_event_number",
            ]
        ).any()
    )

    provenance_pass = bool(
        timeline[
            "score_sources_verified"
        ].all()
        and timeline[
            "reconstruction_verified"
        ].all()
        and (
            ~timeline[
                "prediction_created"
            ]
        ).all()
        and (
            ~timeline[
                "identity_updated"
            ]
        ).all()
        and (
            ~timeline[
                "explanation_created"
            ]
        ).all()
        and (
            ~timeline[
                "future_games_used"
            ]
        ).all()
    )

    audit_pass = bool(
        not timeline.empty
        and event_numbers_sequential
        and home_continuity
        and away_continuity
        and positive_score_changes
        and one_side_scores
        and one_terminal_event
        and terminal_event_is_last
        and final_home_matches
        and final_away_matches
        and event_keys_unique
        and provenance_pass
    )

    return {
        "game_pk":
            game_pk,

        "game_date":
            outcome["game_date"],

        "home_team":
            str(
                outcome["home_team"]
            ),

        "away_team":
            str(
                outcome["away_team"]
            ),

        "timeline_rows":
            int(len(timeline)),

        "timeline_not_empty":
            not timeline.empty,

        "event_numbers_sequential":
            event_numbers_sequential,

        "score_continuity_home":
            home_continuity,

        "score_continuity_away":
            away_continuity,

        "positive_score_changes":
            positive_score_changes,

        "one_side_scores_per_event":
            one_side_scores,

        "one_terminal_event":
            one_terminal_event,

        "terminal_event_is_last":
            terminal_event_is_last,

        "final_home_score_matches":
            final_home_matches,

        "final_away_score_matches":
            final_away_matches,

        "final_score_matches":
            bool(
                final_home_matches
                and final_away_matches
            ),

        "unique_event_keys":
            event_keys_unique,

        "raw_score_state_repairs":
            int(
                timeline[
                    "score_state_repaired"
                ].sum()
            ),

        "within_pa_score_changes":
            int(
                timeline[
                    "score_change_within_plate_appearance"
                ].sum()
            ),

        "tie_events_created":
            int(
                timeline[
                    "tie_created"
                ].sum()
            ),

        "tie_events_broken":
            int(
                timeline[
                    "tie_broken"
                ].sum()
            ),

        "direct_lead_changes":
            int(
                timeline[
                    "direct_lead_change"
                ].sum()
            ),

        "non_tie_leader_changes":
            int(
                timeline[
                    "non_tie_leader_changed"
                ].sum()
            ),

        "provenance_pass":
            provenance_pass,

        "audit_pass":
            audit_pass,
    }


def run_scoring_timeline_season_build(
    season: int = 2024,
    resume: bool = True,
    checkpoint_every: int = 100,
    progress_every: int = 25,
) -> dict[str, Any]:
    started = time.time()
    season = int(season)

    outcome_path = Path(
        str(
            GAME_OUTCOME_TEMPLATE
        ).format(
            season=season
        )
    )

    event_path = Path(
        str(
            EVENT_STORE_TEMPLATE
        ).format(
            season=season
        )
    )

    output_dir = Path(
        str(
            OUTPUT_TEMPLATE
        ).format(
            season=season
        )
    )

    checkpoint_path = (
        output_dir
        / CHECKPOINT_FILENAME
    )

    failure_checkpoint_path = (
        output_dir
        / FAILURE_CHECKPOINT_FILENAME
    )

    final_timeline_path = (
        output_dir
        / FINAL_TIMELINE_FILENAME
    )

    final_audit_path = (
        output_dir
        / FINAL_AUDIT_FILENAME
    )

    final_failure_path = (
        output_dir
        / FINAL_FAILURE_FILENAME
    )

    metadata_path = (
        output_dir
        / METADATA_FILENAME
    )

    if not outcome_path.exists():
        raise FileNotFoundError(
            f"Missing frozen outcomes: "
            f"{outcome_path}"
        )

    if not event_path.exists():
        raise FileNotFoundError(
            f"Missing event store: "
            f"{event_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "ATLAS FULL-SEASON SCORING TIMELINE BUILD"
    )
    print("=" * 80)
    print(
        f"Season.................... "
        f"{season}"
    )
    print(
        "Loading frozen outcomes..."
    )

    outcomes = _normalize_outcomes(
        pd.read_parquet(
            outcome_path
        ),
        season=season,
    )

    print(
        f"Verified games............ "
        f"{len(outcomes):,}"
    )
    print(
        "Loading canonical event store once..."
    )

    available_columns = list(
        pd.read_parquet(
            event_path
        ).columns
    )

    desired_columns = [
        "game_pk",
        "game_date",
        "inning",
        "inning_topbot",
        "outs_when_up",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
        "player_name",
        "events",
        "description",
        "des",
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
        "delta_run_exp",
        "delta_home_win_exp",
    ]

    selected_columns = [
        column
        for column in desired_columns
        if column in available_columns
    ]

    events = _normalize_events(
        pd.read_parquet(
            event_path,
            columns=selected_columns,
        )
    )

    eligible_game_pks = set(
        outcomes[
            "game_pk"
        ].astype("int64")
    )

    events = events[
        events[
            "game_pk"
        ].isin(
            eligible_game_pks
        )
    ].copy()

    print(
        f"Event rows loaded.......... "
        f"{len(events):,}"
    )

    event_groups = {
        int(game_pk):
            group.copy()
        for game_pk, group in events.groupby(
            "game_pk",
            sort=False,
        )
    }

    existing_timeline = pd.DataFrame()
    existing_failures = pd.DataFrame(
        columns=[
            "game_pk",
            "game_date",
            "error_type",
            "error_message",
        ]
    )

    if (
        resume
        and checkpoint_path.exists()
    ):
        existing_timeline = pd.read_parquet(
            checkpoint_path
        )

    if (
        resume
        and failure_checkpoint_path.exists()
    ):
        existing_failures = pd.read_parquet(
            failure_checkpoint_path
        )

    completed_games = set()

    if not existing_timeline.empty:
        completed_games.update(
            pd.to_numeric(
                existing_timeline[
                    "game_pk"
                ],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
            .tolist()
        )

    if not existing_failures.empty:
        completed_games.update(
            pd.to_numeric(
                existing_failures[
                    "game_pk"
                ],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
            .tolist()
        )

    remaining = outcomes[
        ~outcomes[
            "game_pk"
        ].isin(
            completed_games
        )
    ].copy()

    print(
        f"Already checkpointed...... "
        f"{len(completed_games):,}"
    )
    print(
        f"Remaining................. "
        f"{len(remaining):,}"
    )
    print(
        f"Timeline checkpoint....... "
        f"{checkpoint_path}"
    )
    print("=" * 80)

    new_timeline_frames = []
    new_failure_records = []

    run_started = time.time()

    def save_checkpoint() -> None:
        combined_timeline_parts = []

        if not existing_timeline.empty:
            combined_timeline_parts.append(
                existing_timeline
            )

        if new_timeline_frames:
            combined_timeline_parts.extend(
                new_timeline_frames
            )

        if combined_timeline_parts:
            checkpoint_timeline = pd.concat(
                combined_timeline_parts,
                ignore_index=True,
            ).sort_values(
                [
                    "game_date",
                    "game_pk",
                    "scoring_event_number",
                ],
                kind="stable",
            ).reset_index(drop=True)

            _atomic_parquet_write(
                checkpoint_timeline,
                checkpoint_path,
            )

        failure_parts = []

        if not existing_failures.empty:
            failure_parts.append(
                existing_failures
            )

        if new_failure_records:
            failure_parts.append(
                pd.DataFrame(
                    new_failure_records
                )
            )

        if failure_parts:
            checkpoint_failures = pd.concat(
                failure_parts,
                ignore_index=True,
            ).drop_duplicates(
                subset=[
                    "game_pk",
                ],
                keep="last",
            )

        else:
            checkpoint_failures = pd.DataFrame(
                columns=[
                    "game_pk",
                    "game_date",
                    "error_type",
                    "error_message",
                ]
            )

        _atomic_parquet_write(
            checkpoint_failures,
            failure_checkpoint_path,
        )

    for run_index, outcome in enumerate(
        remaining.itertuples(
            index=False
        ),
        start=1,
    ):
        outcome_series = pd.Series(
            outcome._asdict()
        )

        game_pk = int(
            outcome_series["game_pk"]
        )

        try:
            game_events = event_groups.get(
                game_pk,
                pd.DataFrame(),
            )

            timeline = _build_one_game(
                outcome=outcome_series,
                events=game_events,
            )

            new_timeline_frames.append(
                timeline
            )

        except Exception as exc:
            new_failure_records.append({
                "game_pk":
                    game_pk,

                "game_date":
                    outcome_series[
                        "game_date"
                    ],

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            })

        if (
            run_index % checkpoint_every == 0
            or run_index == len(remaining)
        ):
            save_checkpoint()

            completed_total = (
                len(completed_games)
                + run_index
            )

            print(
                f"💾 Checkpoint saved | "
                f"games attempted="
                f"{completed_total:,}/"
                f"{len(outcomes):,}"
            )

        if (
            run_index % progress_every == 0
            or run_index == len(remaining)
        ):
            elapsed = (
                time.time()
                - run_started
            )

            average = (
                elapsed / run_index
                if run_index > 0
                else 0.0
            )

            remaining_games = (
                len(remaining)
                - run_index
            )

            eta_seconds = (
                remaining_games
                * average
            )

            percent = (
                run_index
                / len(remaining)
                * 100
                if len(remaining) > 0
                else 100.0
            )

            print(
                f"Working... "
                f"{run_index:>4,}/"
                f"{len(remaining):,} "
                f"({percent:>5.1f}%) | "
                f"avg={average:>5.2f}s/game | "
                f"ETA={eta_seconds / 60:>6.1f}m"
            )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Timeline checkpoint was not created."
        )

    timeline = pd.read_parquet(
        checkpoint_path
    )

    failures = pd.read_parquet(
        failure_checkpoint_path
    )

    if not timeline.empty:
        timeline["game_pk"] = pd.to_numeric(
            timeline["game_pk"],
            errors="raise",
        ).astype("int64")

        timeline = timeline.sort_values(
            [
                "game_date",
                "game_pk",
                "scoring_event_number",
            ],
            kind="stable",
        ).reset_index(drop=True)

    audit_records = []

    timeline_groups = {
        int(game_pk):
            group.copy()
        for game_pk, group in timeline.groupby(
            "game_pk",
            sort=False,
        )
    }

    for outcome in outcomes.itertuples(
        index=False
    ):
        outcome_series = pd.Series(
            outcome._asdict()
        )

        game_pk = int(
            outcome_series["game_pk"]
        )

        game_timeline = timeline_groups.get(
            game_pk,
            pd.DataFrame(),
        )

        audit_records.append(
            _audit_one_game(
                timeline=game_timeline,
                outcome=outcome_series,
            )
        )

    audit = pd.DataFrame(
        audit_records
    ).sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_event_rows = int(
        timeline.duplicated(
            subset=[
                "game_pk",
                "scoring_event_number",
            ]
        ).sum()
        if not timeline.empty
        else 0
    )

    games_built = int(
        timeline[
            "game_pk"
        ].nunique()
        if not timeline.empty
        else 0
    )

    audit_failures = audit[
        ~audit[
            "audit_pass"
        ]
    ].copy()

    expected_games = int(
        len(outcomes)
    )

    phase_pass = bool(
        games_built == expected_games
        and failures.empty
        and audit_failures.empty
        and duplicate_event_rows == 0
    )

    elapsed = (
        time.time()
        - started
    )

    metadata = {
        "engine":
            "ATLAS Full-Season Scoring Timeline Builder",

        "season_builder_version":
            SEASON_BUILDER_VERSION,

        "scoring_timeline_version":
            SCORING_TIMELINE_VERSION,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "season":
            season,

        "verified_games":
            expected_games,

        "games_built":
            games_built,

        "scoring_state_rows":
            int(len(timeline)),

        "build_failures":
            int(len(failures)),

        "audit_failures":
            int(len(audit_failures)),

        "duplicate_scoring_event_rows":
            duplicate_event_rows,

        "raw_score_state_repairs":
            int(
                timeline[
                    "score_state_repaired"
                ].sum()
                if not timeline.empty
                else 0
            ),

        "within_pa_score_changes":
            int(
                timeline[
                    "score_change_within_plate_appearance"
                ].sum()
                if not timeline.empty
                else 0
            ),

        "phase_2c2_pass":
            phase_pass,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "explanation_created":
            False,

        "future_games_used":
            False,

        "outputs": {
            "scoring_state_timelines":
                str(
                    final_timeline_path
                ),

            "scoring_state_timeline_audit":
                str(
                    final_audit_path
                ),

            "scoring_state_timeline_failures":
                str(
                    final_failure_path
                ),

            "metadata":
                str(
                    metadata_path
                ),

            "checkpoint":
                str(
                    checkpoint_path
                ),
        },

        "elapsed_seconds":
            float(elapsed),

        "built_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    _atomic_parquet_write(
        timeline,
        final_timeline_path,
    )

    _atomic_parquet_write(
        audit,
        final_audit_path,
    )

    _atomic_parquet_write(
        failures,
        final_failure_path,
    )

    _atomic_json_write(
        metadata,
        metadata_path,
    )

    print()
    print("=" * 80)
    print(
        "FULL-SEASON SCORING TIMELINE BUILD COMPLETE"
    )
    print("=" * 80)
    print(
        f"Verified Games............ "
        f"{expected_games:,}"
    )
    print(
        f"Games Built............... "
        f"{games_built:,}"
    )
    print(
        f"Scoring State Rows........ "
        f"{len(timeline):,}"
    )
    print(
        f"Build Failures............ "
        f"{len(failures):,}"
    )
    print(
        f"Audit Failures............ "
        f"{len(audit_failures):,}"
    )
    print(
        f"Duplicate Event Rows...... "
        f"{duplicate_event_rows:,}"
    )
    print(
        f"Raw State Repairs......... "
        f"{metadata['raw_score_state_repairs']:,}"
    )
    print(
        f"Within-PA Score Changes.... "
        f"{metadata['within_pa_score_changes']:,}"
    )
    print(
        f"Phase 2C.2 Pass........... "
        f"{phase_pass}"
    )
    print(
        f"Elapsed................... "
        f"{elapsed / 60:.1f} minutes"
    )
    print("=" * 80)

    return {
        "timeline":
            timeline,

        "audit":
            audit,

        "failures":
            failures,

        "metadata":
            metadata,
    }
