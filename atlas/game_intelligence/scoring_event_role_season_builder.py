"""
Full-season scoring-event role builder for Project ATLAS.

Consumes the frozen Phase 2C scoring timeline and attaches factual
Phase 2D scoring-event roles.

This module creates no predictions, identities, explanations,
sportsbook features, or pregame features.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import time

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION
from .scoring_event_roles import (
    SCORING_EVENT_ROLE_VERSION,
    classify_scoring_timeline,
    attach_decisive_scoring_flags,
)


SCORING_EVENT_ROLE_SEASON_VERSION = "1.0.0"

REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT = REPO_ROOT / "data"

TIMELINE_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "scoring_timelines"
    / "{season}"
    / "scoring_state_timelines.parquet"
)

OUTCOME_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "outcomes"
    / "{season}"
    / "game_outcomes.parquet"
)

OUTPUT_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "scoring_event_roles"
    / "{season}"
)

ROLE_FILENAME = (
    "scoring_event_roles.parquet"
)

AUDIT_FILENAME = (
    "scoring_event_role_audit.parquet"
)

FAILURE_FILENAME = (
    "scoring_event_role_failures.parquet"
)

METADATA_FILENAME = (
    "scoring_event_role_metadata.json"
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


def _normalize_timeline(
    timeline: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "scoring_event_number",
        "scoring_team",
        "scoring_side",
        "batting_side",
        "inning",
        "pre_home_score",
        "pre_away_score",
        "post_home_score",
        "post_away_score",
        "runs_on_play",
        "terminal_scoring_event",
        "score_sources_verified",
        "reconstruction_verified",
    }

    missing = required - set(
        timeline.columns
    )

    if missing:
        raise KeyError(
            "Frozen scoring timeline missing columns: "
            f"{sorted(missing)}"
        )

    normalized = timeline.copy()

    integer_columns = [
        "game_pk",
        "atlas_season",
        "scoring_event_number",
        "inning",
        "pre_home_score",
        "pre_away_score",
        "post_home_score",
        "post_away_score",
        "runs_on_play",
    ]

    for column in integer_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="raise",
        ).astype("int64")

    normalized["game_date"] = pd.to_datetime(
        normalized["game_date"],
        errors="raise",
    ).dt.normalize()

    normalized = normalized[
        normalized["atlas_season"].eq(
            int(season)
        )
    ].copy()

    if normalized.duplicated(
        subset=[
            "game_pk",
            "scoring_event_number",
        ]
    ).any():
        raise AssertionError(
            "Frozen timeline contains duplicate scoring events."
        )

    if not normalized[
        "score_sources_verified"
    ].fillna(False).all():
        raise AssertionError(
            "Timeline contains unverified score sources."
        )

    if not normalized[
        "reconstruction_verified"
    ].fillna(False).all():
        raise AssertionError(
            "Timeline contains unverified reconstructions."
        )

    if not normalized[
        "batting_side"
    ].eq(
        normalized[
            "scoring_side"
        ]
    ).all():
        raise AssertionError(
            "Canonical batting and scoring sides differ."
        )

    return normalized.sort_values(
        [
            "game_date",
            "game_pk",
            "scoring_event_number",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _normalize_outcomes(
    outcomes: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "atlas_season",
        "winner_team",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing = required - set(
        outcomes.columns
    )

    if missing:
        raise KeyError(
            "Frozen outcomes missing columns: "
            f"{sorted(missing)}"
        )

    normalized = outcomes.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    normalized = normalized[
        normalized["atlas_season"].eq(
            int(season)
        )
    ].copy()

    if normalized["game_pk"].duplicated().any():
        raise AssertionError(
            "Frozen outcomes contain duplicate games."
        )

    return normalized.reset_index(
        drop=True
    )


def _audit_one_game(
    game: pd.DataFrame,
) -> dict[str, Any]:
    game = game.sort_values(
        "scoring_event_number",
        kind="stable",
    )

    game_pk = int(
        game["game_pk"].iloc[0]
    )

    role_columns = [
        "opening_score",
        "tying_score",
        "go_ahead_score",
        "lead_extension",
        "deficit_reduction",
    ]

    event_numbers_sequential = bool(
        game[
            "scoring_event_number"
        ].tolist()
        == list(
            range(
                1,
                len(game) + 1,
            )
        )
    )

    one_primary_role_per_event = bool(
        game[
            role_columns
        ].sum(
            axis=1
        ).eq(1).all()
    )

    role_name_matches_boolean = bool(
        all(
            bool(
                row[
                    row[
                        "primary_scoring_role"
                    ]
                ]
            )
            for _, row in game.iterrows()
        )
    )

    one_decisive_event = bool(
        int(
            game[
                "decisive_scoring_event"
            ].sum()
        ) == 1
    )

    decisive = game[
        game[
            "decisive_scoring_event"
        ]
    ]

    decisive_by_winner = bool(
        len(decisive) == 1
        and decisive[
            "scoring_team"
        ].iloc[0]
        == game[
            "winner_team"
        ].iloc[0]
    )

    exactly_one_opening_score = bool(
        int(
            game[
                "opening_score"
            ].sum()
        ) == 1
    )

    first_event_is_opening_score = bool(
        game[
            "opening_score"
        ].iloc[0]
    )

    final_event_matches_terminal = bool(
        int(
            game[
                "terminal_scoring_event_role"
            ].sum()
        ) == 1
        and bool(
            game[
                "terminal_scoring_event_role"
            ].iloc[-1]
        )
    )

    margin_math_pass = bool(
        game[
            "pre_scoring_team_margin"
        ].eq(
            game[
                "pre_scoring_team_score"
            ]
            - game[
                "pre_opponent_score"
            ]
        ).all()
        and game[
            "post_scoring_team_margin"
        ].eq(
            game[
                "post_scoring_team_score"
            ]
            - game[
                "post_opponent_score"
            ]
        ).all()
    )

    provenance_pass = bool(
        game[
            "postgame_hindsight_only"
        ].all()
        and (
            ~game[
                "prediction_created"
            ]
        ).all()
        and (
            ~game[
                "identity_updated"
            ]
        ).all()
        and (
            ~game[
                "explanation_created"
            ]
        ).all()
        and (
            ~game[
                "future_games_used"
            ]
        ).all()
    )

    audit_pass = bool(
        event_numbers_sequential
        and one_primary_role_per_event
        and role_name_matches_boolean
        and one_decisive_event
        and decisive_by_winner
        and exactly_one_opening_score
        and first_event_is_opening_score
        and final_event_matches_terminal
        and margin_math_pass
        and provenance_pass
    )

    return {
        "game_pk":
            game_pk,

        "game_date":
            game[
                "game_date"
            ].iloc[0],

        "home_team":
            game[
                "home_team"
            ].iloc[0],

        "away_team":
            game[
                "away_team"
            ].iloc[0],

        "winner_team":
            game[
                "winner_team"
            ].iloc[0],

        "scoring_event_rows":
            int(len(game)),

        "event_numbers_sequential":
            event_numbers_sequential,

        "one_primary_role_per_event":
            one_primary_role_per_event,

        "role_name_matches_boolean":
            role_name_matches_boolean,

        "one_decisive_event":
            one_decisive_event,

        "decisive_event_by_winner":
            decisive_by_winner,

        "exactly_one_opening_score":
            exactly_one_opening_score,

        "first_event_is_opening_score":
            first_event_is_opening_score,

        "one_terminal_role":
            final_event_matches_terminal,

        "margin_math_pass":
            margin_math_pass,

        "provenance_pass":
            provenance_pass,

        "opening_scores":
            int(
                game[
                    "opening_score"
                ].sum()
            ),

        "tying_scores":
            int(
                game[
                    "tying_score"
                ].sum()
            ),

        "go_ahead_scores":
            int(
                game[
                    "go_ahead_score"
                ].sum()
            ),

        "lead_extensions":
            int(
                game[
                    "lead_extension"
                ].sum()
            ),

        "deficit_reductions":
            int(
                game[
                    "deficit_reduction"
                ].sum()
            ),

        "decisive_scoring_events":
            int(
                game[
                    "decisive_scoring_event"
                ].sum()
            ),

        "audit_pass":
            audit_pass,
    }


def run_scoring_event_role_season_build(
    season: int = 2024,
) -> dict[str, Any]:
    started = time.time()
    season = int(season)

    timeline_path = Path(
        str(
            TIMELINE_TEMPLATE
        ).format(
            season=season
        )
    )

    outcome_path = Path(
        str(
            OUTCOME_TEMPLATE
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

    role_path = (
        output_dir
        / ROLE_FILENAME
    )

    audit_path = (
        output_dir
        / AUDIT_FILENAME
    )

    failure_path = (
        output_dir
        / FAILURE_FILENAME
    )

    metadata_path = (
        output_dir
        / METADATA_FILENAME
    )

    if not timeline_path.exists():
        raise FileNotFoundError(
            f"Missing frozen timeline: {timeline_path}"
        )

    if not outcome_path.exists():
        raise FileNotFoundError(
            f"Missing frozen outcomes: {outcome_path}"
        )

    print("=" * 82)
    print(
        "ATLAS FULL-SEASON SCORING-EVENT ROLE BUILD"
    )
    print("=" * 82)
    print(
        f"Season...................... "
        f"{season}"
    )

    timeline = _normalize_timeline(
        pd.read_parquet(
            timeline_path
        ),
        season=season,
    )

    outcomes = _normalize_outcomes(
        pd.read_parquet(
            outcome_path
        ),
        season=season,
    )

    print(
        f"Frozen timeline rows........ "
        f"{len(timeline):,}"
    )
    print(
        f"Verified games.............. "
        f"{timeline['game_pk'].nunique():,}"
    )

    failure_records = []

    try:
        roles = classify_scoring_timeline(
            timeline
        )

        roles = attach_decisive_scoring_flags(
            classified=roles,
            outcomes=outcomes,
        )

    except Exception as exc:
        failure_records.append({
            "season":
                season,

            "error_type":
                type(exc).__name__,

            "error_message":
                str(exc),
        })

        roles = pd.DataFrame()

    failures = pd.DataFrame(
        failure_records,
        columns=[
            "season",
            "error_type",
            "error_message",
        ],
    )

    audit_records = []

    if not roles.empty:
        game_groups = roles.groupby(
            "game_pk",
            sort=False,
        )

        for count, (
            game_pk,
            game,
        ) in enumerate(
            game_groups,
            start=1,
        ):
            audit_records.append(
                _audit_one_game(
                    game
                )
            )

            if (
                count % 500 == 0
                or count
                == roles[
                    "game_pk"
                ].nunique()
            ):
                print(
                    f"Audited "
                    f"{count:>4,}/"
                    f"{roles['game_pk'].nunique():,} "
                    f"games"
                )

    audit = pd.DataFrame(
        audit_records
    )

    if not audit.empty:
        audit = audit.sort_values(
            [
                "game_date",
                "game_pk",
            ],
            kind="stable",
        ).reset_index(drop=True)

    duplicate_rows = int(
        roles.duplicated(
            subset=[
                "game_pk",
                "scoring_event_number",
            ]
        ).sum()
        if not roles.empty
        else 0
    )

    games_built = int(
        roles[
            "game_pk"
        ].nunique()
        if not roles.empty
        else 0
    )

    audit_failures = (
        audit[
            ~audit[
                "audit_pass"
            ]
        ].copy()
        if not audit.empty
        else pd.DataFrame()
    )

    expected_rows = int(
        len(timeline)
    )

    expected_games = int(
        timeline[
            "game_pk"
        ].nunique()
    )

    phase_pass = bool(
        len(roles) == expected_rows
        and games_built == expected_games
        and failures.empty
        and audit_failures.empty
        and duplicate_rows == 0
    )

    elapsed = (
        time.time()
        - started
    )

    metadata = {
        "engine":
            "ATLAS Full-Season Scoring-Event Role Builder",

        "season":
            season,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "scoring_event_role_version":
            SCORING_EVENT_ROLE_VERSION,

        "season_builder_version":
            SCORING_EVENT_ROLE_SEASON_VERSION,

        "source_timeline_rows":
            expected_rows,

        "role_rows":
            int(len(roles)),

        "verified_games":
            expected_games,

        "games_built":
            games_built,

        "build_failures":
            int(len(failures)),

        "audit_failures":
            int(len(audit_failures)),

        "duplicate_role_rows":
            duplicate_rows,

        "opening_scores":
            int(
                roles[
                    "opening_score"
                ].sum()
                if not roles.empty
                else 0
            ),

        "tying_scores":
            int(
                roles[
                    "tying_score"
                ].sum()
                if not roles.empty
                else 0
            ),

        "go_ahead_scores":
            int(
                roles[
                    "go_ahead_score"
                ].sum()
                if not roles.empty
                else 0
            ),

        "lead_extensions":
            int(
                roles[
                    "lead_extension"
                ].sum()
                if not roles.empty
                else 0
            ),

        "deficit_reductions":
            int(
                roles[
                    "deficit_reduction"
                ].sum()
                if not roles.empty
                else 0
            ),

        "decisive_scoring_events":
            int(
                roles[
                    "decisive_scoring_event"
                ].sum()
                if not roles.empty
                else 0
            ),

        "phase_2d2_pass":
            phase_pass,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "explanation_created":
            False,

        "future_games_used":
            False,

        "elapsed_seconds":
            float(elapsed),

        "built_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": {
            "roles":
                str(role_path),

            "audit":
                str(audit_path),

            "failures":
                str(failure_path),

            "metadata":
                str(metadata_path),
        },
    }

    _atomic_parquet_write(
        roles,
        role_path,
    )

    _atomic_parquet_write(
        audit,
        audit_path,
    )

    _atomic_parquet_write(
        failures,
        failure_path,
    )

    _atomic_json_write(
        metadata,
        metadata_path,
    )

    print()
    print("=" * 82)
    print(
        "FULL-SEASON SCORING-EVENT ROLE BUILD COMPLETE"
    )
    print("=" * 82)
    print(
        f"Source Timeline Rows........ "
        f"{expected_rows:,}"
    )
    print(
        f"Role Rows................... "
        f"{len(roles):,}"
    )
    print(
        f"Games Built................. "
        f"{games_built:,}"
    )
    print(
        f"Build Failures.............. "
        f"{len(failures):,}"
    )
    print(
        f"Audit Failures.............. "
        f"{len(audit_failures):,}"
    )
    print(
        f"Duplicate Role Rows......... "
        f"{duplicate_rows:,}"
    )
    print(
        f"Decisive Events............. "
        f"{metadata['decisive_scoring_events']:,}"
    )
    print(
        f"Phase 2D.2 Pass............. "
        f"{phase_pass}"
    )
    print(
        f"Elapsed..................... "
        f"{elapsed:.1f} seconds"
    )
    print("=" * 82)

    return {
        "roles":
            roles,

        "audit":
            audit,

        "failures":
            failures,

        "metadata":
            metadata,
    }
