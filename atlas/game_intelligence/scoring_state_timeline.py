"""
Canonical scoring-state timeline for Project ATLAS.

This module creates one immutable state row after every scoring
plate appearance in a verified game.

The timeline is factual only. It does not:

- explain why a game occurred
- update team or player identities
- discover evidence or concepts
- create predictions
- use sportsbook information
- use future games
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION


SCORING_TIMELINE_VERSION = "1.0.0"

REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT = REPO_ROOT / "data"

GAME_OUTCOME_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "outcomes"
    / "2024"
    / "game_outcomes.parquet"
)

EVENT_STORE_TEMPLATE = (
    DATA_ROOT
    / "history"
    / "game_cards"
    / "events"
    / "game_events_{season}_regular.parquet"
)


def _load_verified_outcome(
    game_pk: int,
    season: int,
) -> pd.Series:
    """
    Load the frozen Phase 2A outcome for one game.
    """
    if not GAME_OUTCOME_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen game outcomes: "
            f"{GAME_OUTCOME_PATH}"
        )

    outcomes = pd.read_parquet(
        GAME_OUTCOME_PATH
    )

    outcomes["game_pk"] = pd.to_numeric(
        outcomes["game_pk"],
        errors="raise",
    ).astype("int64")

    outcomes["atlas_season"] = pd.to_numeric(
        outcomes["atlas_season"],
        errors="raise",
    ).astype("int64")

    game_rows = outcomes[
        outcomes["game_pk"].eq(
            int(game_pk)
        )
        & outcomes["atlas_season"].eq(
            int(season)
        )
    ]

    if len(game_rows) != 1:
        raise ValueError(
            f"Expected one frozen outcome row for "
            f"game {game_pk}, found {len(game_rows)}."
        )

    outcome = game_rows.iloc[0]

    if not bool(
        outcome["reconstruction_verified"]
    ):
        raise ValueError(
            f"Game {game_pk} is not reconstruction verified."
        )

    if not bool(
        outcome["score_sources_verified"]
    ):
        raise ValueError(
            f"Game {game_pk} does not have verified scores."
        )

    return outcome


def _load_game_events(
    game_pk: int,
    season: int,
) -> pd.DataFrame:
    """
    Load canonical pitch events for one game.
    """
    event_path = Path(
        str(EVENT_STORE_TEMPLATE).format(
            season=int(season)
        )
    )

    if not event_path.exists():
        raise FileNotFoundError(
            f"Missing event store: {event_path}"
        )

    required_columns = [
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

    available_columns = list(
        pd.read_parquet(
            event_path,
        ).columns
    )

    selected_columns = [
        column
        for column in required_columns
        if column in available_columns
    ]

    events = pd.read_parquet(
        event_path,
        columns=selected_columns,
    )

    events["game_pk"] = pd.to_numeric(
        events["game_pk"],
        errors="coerce",
    )

    events = events[
        events["game_pk"].eq(
            int(game_pk)
        )
    ].copy()

    if events.empty:
        raise ValueError(
            f"No event rows found for game {game_pk}."
        )

    required_event_columns = {
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

    missing = (
        required_event_columns
        - set(events.columns)
    )

    if missing:
        raise KeyError(
            "Event store missing required columns: "
            f"{sorted(missing)}"
        )

    events["_source_order"] = np.arange(
        len(events),
        dtype="int64",
    )

    numeric_columns = [
        "inning",
        "outs_when_up",
        "at_bat_number",
        "pitch_number",
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
        "delta_run_exp",
        "delta_home_win_exp",
    ]

    for column in numeric_columns:
        if column in events.columns:
            events[column] = pd.to_numeric(
                events[column],
                errors="coerce",
            )

    events = events.sort_values(
        [
            "at_bat_number",
            "pitch_number",
            "_source_order",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return events


def _leader(
    home_score: int,
    away_score: int,
) -> str:
    """
    Return HOME, AWAY or TIE for one score state.
    """
    if home_score > away_score:
        return "HOME"

    if away_score > home_score:
        return "AWAY"

    return "TIE"


def _lead_size(
    home_score: int,
    away_score: int,
) -> int:
    """
    Absolute score lead.
    """
    return abs(
        int(home_score)
        - int(away_score)
    )


def _terminal_plate_appearances(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce pitch rows to the final row of each plate appearance.
    """
    terminal = (
        events.groupby(
            "at_bat_number",
            sort=False,
            as_index=False,
        )
        .tail(1)
        .sort_values(
            [
                "at_bat_number",
                "pitch_number",
                "_source_order",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    score_columns = [
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
    ]

    if terminal[
        score_columns
    ].isna().any().any():
        raise AssertionError(
            "Terminal plate appearances contain "
            "missing score states."
        )

    for column in score_columns:
        terminal[column] = (
            terminal[column]
            .astype("int64")
        )

    terminal["home_runs_on_play"] = (
        terminal["post_home_score"]
        - terminal["home_score"]
    )

    terminal["away_runs_on_play"] = (
        terminal["post_away_score"]
        - terminal["away_score"]
    )

    terminal["runs_on_play"] = (
        terminal["home_runs_on_play"]
        + terminal["away_runs_on_play"]
    )

    if terminal[
        [
            "home_runs_on_play",
            "away_runs_on_play",
        ]
    ].lt(0).any().any():
        raise AssertionError(
            "A terminal plate appearance reduced the score."
        )

    return terminal



def _score_change_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract every ordered event row that changes the score.

    A run may score before a plate appearance terminates, including:

    - wild pitches
    - passed balls
    - balks
    - steals of home
    - errors
    - other runner-advance events

    Therefore, terminal plate appearances alone are not a complete
    scoring ledger.
    """
    if events.empty:
        raise ValueError(
            "Cannot extract score changes from empty events."
        )

    ordered = events.sort_values(
        [
            "at_bat_number",
            "pitch_number",
            "_source_order",
        ],
        kind="stable",
    ).reset_index(drop=True)

    required = [
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
    ]

    if ordered[
        required
    ].isna().any().any():
        raise AssertionError(
            "Event rows contain missing score states."
        )

    for column in required:
        ordered[column] = (
            pd.to_numeric(
                ordered[column],
                errors="raise",
            )
            .astype("int64")
        )

    canonical_home_score = int(
        ordered[
            "home_score"
        ].iloc[0]
    )

    canonical_away_score = int(
        ordered[
            "away_score"
        ].iloc[0]
    )

    scoring_records = []

    for row in ordered.itertuples(
        index=False
    ):
        raw_pre_home_score = int(
            row.home_score
        )

        raw_pre_away_score = int(
            row.away_score
        )

        post_home_score = int(
            row.post_home_score
        )

        post_away_score = int(
            row.post_away_score
        )

        home_delta = int(
            post_home_score
            - canonical_home_score
        )

        away_delta = int(
            post_away_score
            - canonical_away_score
        )

        # Repeated pitch rows commonly retain the same post-score.
        if (
            home_delta == 0
            and away_delta == 0
        ):
            continue

        if (
            home_delta < 0
            or away_delta < 0
        ):
            raise AssertionError(
                "Canonical score decreased while scanning "
                "ordered event rows."
            )

        if (
            home_delta > 0
            and away_delta > 0
        ):
            raise AssertionError(
                "Both teams gained runs on one ordered event row."
            )

        record = row._asdict()

        record[
            "canonical_pre_home_score"
        ] = canonical_home_score

        record[
            "canonical_pre_away_score"
        ] = canonical_away_score

        record[
            "raw_pre_home_score"
        ] = raw_pre_home_score

        record[
            "raw_pre_away_score"
        ] = raw_pre_away_score

        record[
            "raw_pre_score_matches_canonical"
        ] = bool(
            raw_pre_home_score
            == canonical_home_score
            and raw_pre_away_score
            == canonical_away_score
        )

        record[
            "score_state_repaired"
        ] = bool(
            not record[
                "raw_pre_score_matches_canonical"
            ]
        )

        record[
            "home_runs_on_play"
        ] = home_delta

        record[
            "away_runs_on_play"
        ] = away_delta

        record[
            "runs_on_play"
        ] = int(
            home_delta
            + away_delta
        )

        scoring_records.append(
            record
        )

        canonical_home_score = (
            post_home_score
        )

        canonical_away_score = (
            post_away_score
        )

    scoring = pd.DataFrame(
        scoring_records
    )

    if scoring.empty:
        raise AssertionError(
            "No score-changing event rows were found."
        )

    scoring[
        "score_change_within_plate_appearance"
    ] = (
        scoring.duplicated(
            subset=[
                "at_bat_number",
            ],
            keep=False,
        )
    )

    return scoring.reset_index(
        drop=True
    )




def _canonical_scoring_attribution(
    *,
    source_inning: int,
    source_inning_half: str,
    pitch_number: int,
    scoring_side: str,
    score_state_repaired: bool,
) -> dict[str, object]:
    """
    Determine the canonical inning-half attribution of a score change.

    Statcast can occasionally delay a score update until the first
    event row of the following half-inning. In that case, the raw
    source row belongs to one batting side while the score increase
    belongs to the opponent.

    The source location is always preserved. Only the canonical
    attribution is repaired.
    """
    source_inning = int(
        source_inning
    )

    source_half = str(
        source_inning_half
    )

    source_side = (
        "AWAY"
        if source_half.lower().startswith(
            "top"
        )
        else "HOME"
    )

    scoring_side = str(
        scoring_side
    )

    delayed_score_update = bool(
        score_state_repaired
        and int(pitch_number) == 1
        and source_side != scoring_side
    )

    if not delayed_score_update:
        return {
            "source_inning":
                source_inning,

            "source_inning_half":
                source_half,

            "source_batting_side":
                source_side,

            "inning":
                source_inning,

            "inning_half":
                source_half,

            "batting_side":
                source_side,

            "delayed_score_update":
                False,

            "scoring_attribution_repaired":
                False,
        }

    # Away-team run first appears in the bottom half:
    # attribute it to the top of the same inning.
    if (
        source_side == "HOME"
        and scoring_side == "AWAY"
    ):
        attributed_inning = (
            source_inning
        )

        attributed_half = "Top"

    # Home-team run first appears in the next top half:
    # attribute it to the bottom of the previous inning.
    elif (
        source_side == "AWAY"
        and scoring_side == "HOME"
    ):
        attributed_inning = max(
            1,
            source_inning - 1,
        )

        attributed_half = "Bot"

    else:
        raise AssertionError(
            "Unsupported delayed score-update attribution."
        )

    return {
        "source_inning":
            source_inning,

        "source_inning_half":
            source_half,

        "source_batting_side":
            source_side,

        "inning":
            int(
                attributed_inning
            ),

        "inning_half":
            attributed_half,

        "batting_side":
            scoring_side,

        "delayed_score_update":
            True,

        "scoring_attribution_repaired":
            True,
    }


def build_scoring_state_timeline(
    game_pk: int,
    season: int = 2024,
) -> pd.DataFrame:
    """
    Build one state row after every scoring plate appearance.

    The resulting timeline is the canonical Phase 2C source for
    later lead, momentum and anatomy summaries.
    """
    game_pk = int(game_pk)
    season = int(season)

    outcome = _load_verified_outcome(
        game_pk=game_pk,
        season=season,
    )

    events = _load_game_events(
        game_pk=game_pk,
        season=season,
    )

    scoring = _score_change_events(
        events
    )

    if scoring.empty:
        raise AssertionError(
            f"Verified completed game {game_pk} "
            "contains no score-changing event rows."
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

    previous_non_tie_leader: str | None = None

    # Statcast's raw pre-play score fields are retained for
    # provenance, but canonical continuity is carried forward
    # from the previous verified scoring state.
    canonical_home_score = int(
        scoring[
            "canonical_pre_home_score"
        ].iloc[0]
    )

    canonical_away_score = int(
        scoring[
            "canonical_pre_away_score"
        ].iloc[0]
    )

    for event_number, row in enumerate(
        scoring.itertuples(
            index=False,
        ),
        start=1,
    ):
        raw_pre_home_score = int(
            row.home_score
        )

        raw_pre_away_score = int(
            row.away_score
        )

        pre_home_score = int(
            canonical_home_score
        )

        pre_away_score = int(
            canonical_away_score
        )

        post_home_score = int(
            row.post_home_score
        )

        post_away_score = int(
            row.post_away_score
        )

        home_runs = int(
            post_home_score
            - pre_home_score
        )

        away_runs = int(
            post_away_score
            - pre_away_score
        )

        runs_on_play = int(
            home_runs
            + away_runs
        )

        raw_pre_score_matches_canonical = bool(
            raw_pre_home_score
            == pre_home_score
            and raw_pre_away_score
            == pre_away_score
        )

        score_state_repaired = bool(
            not raw_pre_score_matches_canonical
        )

        if home_runs < 0 or away_runs < 0:
            raise AssertionError(
                f"Canonical score decreased during scoring event "
                f"{event_number} in game {game_pk}."
            )

        if runs_on_play <= 0:
            raise AssertionError(
                f"Scoring event {event_number} in game {game_pk} "
                "did not increase the canonical score."
            )

        if home_runs > 0 and away_runs > 0:
            raise AssertionError(
                f"Both teams scored on one plate appearance in "
                f"game {game_pk}, event {event_number}."
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

        tie_created = bool(
            pre_leader != "TIE"
            and post_leader == "TIE"
        )

        tie_broken = bool(
            pre_leader == "TIE"
            and post_leader != "TIE"
        )

        direct_lead_change = bool(
            pre_leader in {
                "HOME",
                "AWAY",
            }
            and post_leader in {
                "HOME",
                "AWAY",
            }
            and pre_leader != post_leader
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

        event_result = (
            getattr(
                row,
                "events",
                None,
            )
        )

        play_description = (
            getattr(
                row,
                "des",
                None,
            )
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

            "terminal_pitch_number":
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
                else int(row.batter)
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
                else int(row.pitcher)
            ),

            "pitcher_name":
                getattr(
                    row,
                    "player_name",
                    None,
                ),

            "event_result":
                event_result,

            "play_description":
                play_description,

            "raw_pre_home_score":
                raw_pre_home_score,

            "raw_pre_away_score":
                raw_pre_away_score,

            "pre_home_score":
                pre_home_score,

            "pre_away_score":
                pre_away_score,

            "raw_pre_score_matches_canonical":
                raw_pre_score_matches_canonical,

            "score_state_repaired":
                score_state_repaired,

            "score_change_within_plate_appearance":
                bool(
                    getattr(
                        row,
                        "score_change_within_plate_appearance",
                        False,
                    )
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
                tie_created,

            "tie_broken":
                tie_broken,

            "direct_lead_change":
                direct_lead_change,

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
        })

        canonical_home_score = (
            post_home_score
        )

        canonical_away_score = (
            post_away_score
        )

    timeline = pd.DataFrame(
        records
    )

    timeline.loc[
        timeline.index[-1],
        "terminal_scoring_event",
    ] = True

    final_row = timeline.iloc[-1]

    final_score_matches = bool(
        int(
            final_row[
                "post_home_score"
            ]
        )
        == int(
            outcome["home_score"]
        )
        and int(
            final_row[
                "post_away_score"
            ]
        )
        == int(
            outcome["away_score"]
        )
    )

    if not final_score_matches:
        raise AssertionError(
            f"Timeline final score for game {game_pk} "
            "does not match the frozen outcome."
        )

    score_continuity = bool(
        (
            timeline[
                "pre_home_score"
            ].iloc[1:].to_numpy()
            == timeline[
                "post_home_score"
            ].iloc[:-1].to_numpy()
        ).all()
        and (
            timeline[
                "pre_away_score"
            ].iloc[1:].to_numpy()
            == timeline[
                "post_away_score"
            ].iloc[:-1].to_numpy()
        ).all()
    )

    if not score_continuity:
        raise AssertionError(
            f"Scoring timeline for game {game_pk} "
            "does not have continuous score states."
        )

    if int(
        timeline[
            "terminal_scoring_event"
        ].sum()
    ) != 1:
        raise AssertionError(
            "Exactly one terminal scoring event is required."
        )

    return timeline.reset_index(
        drop=True
    )


def summarize_scoring_timeline(
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a compact factual summary of one scoring timeline.
    """
    if timeline.empty:
        raise ValueError(
            "Cannot summarize an empty scoring timeline."
        )

    if timeline[
        "game_pk"
    ].nunique() != 1:
        raise ValueError(
            "Timeline summary requires exactly one game."
        )

    first = timeline.iloc[0]
    last = timeline.iloc[-1]

    scoring_innings = sorted(
        timeline["inning"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    home_scoring_events = int(
        timeline[
            "scoring_side"
        ].eq("HOME").sum()
    )

    away_scoring_events = int(
        timeline[
            "scoring_side"
        ].eq("AWAY").sum()
    )

    summary = pd.DataFrame([
        {
            "game_pk":
                int(first["game_pk"]),

            "game_date":
                first["game_date"],

            "atlas_season":
                int(
                    first[
                        "atlas_season"
                    ]
                ),

            "home_team":
                str(
                    first[
                        "home_team"
                    ]
                ),

            "away_team":
                str(
                    first[
                        "away_team"
                    ]
                ),

            "scoring_events":
                int(len(timeline)),

            "home_scoring_events":
                home_scoring_events,

            "away_scoring_events":
                away_scoring_events,

            "first_scoring_team":
                str(
                    first[
                        "scoring_team"
                    ]
                ),

            "first_scoring_side":
                str(
                    first[
                        "scoring_side"
                    ]
                ),

            "first_scoring_inning":
                int(first["inning"]),

            "first_scoring_half":
                str(
                    first[
                        "inning_half"
                    ]
                ),

            "last_scoring_team":
                str(
                    last[
                        "scoring_team"
                    ]
                ),

            "last_scoring_side":
                str(
                    last[
                        "scoring_side"
                    ]
                ),

            "last_scoring_inning":
                int(last["inning"]),

            "last_scoring_half":
                str(
                    last[
                        "inning_half"
                    ]
                ),

            "terminal_runs_scored":
                int(
                    last[
                        "runs_on_play"
                    ]
                ),

            "scoring_innings":
                scoring_innings,

            "scoring_inning_count":
                int(
                    len(
                        scoring_innings
                    )
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

            "final_home_score":
                int(
                    last[
                        "post_home_score"
                    ]
                ),

            "final_away_score":
                int(
                    last[
                        "post_away_score"
                    ]
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
        }
    ])

    return summary
