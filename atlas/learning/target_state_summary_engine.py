
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
LEARNING_SEASON = 2024

STATE_ROOT = (
    DATA_DIR
    / "learning"
    / "baseball_states"
    / str(LEARNING_SEASON)
)

CHECKPOINT_DIR = (
    STATE_ROOT
    / "team_checkpoints"
)

MASTER_ACTIVE_PATH = (
    STATE_ROOT
    / "team_game_active_concepts.parquet"
)

TARGET_STATE_PATH = (
    STATE_ROOT
    / "team_game_target_states.parquet"
)

METADATA_PATH = (
    STATE_ROOT
    / "target_state_summary_metadata.json"
)


TARGETS = [
    "won",
    "lost",
    "team_scored_5_plus",
    "team_scored_3_or_less",
    "team_scored_8_plus",
    "team_allowed_3_or_less",
    "team_allowed_5_plus",
    "game_total_10_5_plus",
    "game_total_12_plus",
    "game_total_15_plus",
    "game_total_17_plus",
]


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

    temporary.replace(destination)


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary.replace(destination)


def _load_team_active(
    team: str,
) -> pd.DataFrame:
    path = (
        CHECKPOINT_DIR
        / f"{team}_active_concepts.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing active-concept checkpoint: {path}"
        )

    active = pd.read_parquet(path)

    active["game_date"] = pd.to_datetime(
        active["game_date"],
        errors="raise",
    ).dt.normalize()

    return active


def _load_team_games(
    team: str,
) -> pd.DataFrame:
    path = (
        CHECKPOINT_DIR
        / f"{team}_state_summary.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing team-state checkpoint: {path}"
        )

    games = pd.read_parquet(path)

    games["game_date"] = pd.to_datetime(
        games["game_date"],
        errors="raise",
    ).dt.normalize()

    columns = [
        column
        for column in [
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
            "opponent",
            "home_away",
            "opposing_starting_pitcher_id",
            "complete_snapshot_join",
            "strict_backtest_safe",
        ]
        if column in games.columns
    ]

    return games[columns].copy()


def _build_team_target_states(
    team: str,
) -> pd.DataFrame:
    active = _load_team_active(team)
    games = _load_team_games(team)

    target_frame = pd.DataFrame({
        "target": TARGETS
    })

    games["_join_key"] = 1
    target_frame["_join_key"] = 1

    full_grid = games.merge(
        target_frame,
        on="_join_key",
        how="inner",
    ).drop(columns="_join_key")

    if active.empty:
        aggregates = pd.DataFrame(
            columns=[
                "game_pk",
                "team",
                "target",
            ]
        )

    else:
        active = active.copy()

        active["signed_confidence_lift"] = (
            pd.to_numeric(
                active["concept_confidence_score"],
                errors="coerce",
            ).fillna(0.0)
            *
            pd.to_numeric(
                active["weighted_lift"],
                errors="coerce",
            ).fillna(0.0)
        )

        active["support_score"] = np.where(
            active["effect_direction"].eq(
                "supports_target"
            ),
            active["signed_confidence_lift"].abs(),
            0.0,
        )

        active["suppression_score"] = np.where(
            active["effect_direction"].eq(
                "suppresses_target"
            ),
            active["signed_confidence_lift"].abs(),
            0.0,
        )

        aggregates = (
            active.groupby(
                [
                    "game_pk",
                    "team",
                    "target",
                ],
                sort=False,
            )
            .agg(
                active_concepts=(
                    "concept_id",
                    "nunique",
                ),
                active_strong_concepts=(
                    "concept_lifecycle_status",
                    lambda values: int(
                        values.eq(
                            "strong_concept_candidate"
                        ).sum()
                    ),
                ),
                active_candidate_concepts=(
                    "concept_lifecycle_status",
                    lambda values: int(
                        values.eq(
                            "concept_candidate"
                        ).sum()
                    ),
                ),
                supporting_concepts=(
                    "effect_direction",
                    lambda values: int(
                        values.eq(
                            "supports_target"
                        ).sum()
                    ),
                ),
                suppressing_concepts=(
                    "effect_direction",
                    lambda values: int(
                        values.eq(
                            "suppresses_target"
                        ).sum()
                    ),
                ),
                unique_domains=(
                    "concept_domain",
                    "nunique",
                ),
                unique_concept_names=(
                    "concept_name",
                    "nunique",
                ),
                mean_concept_confidence=(
                    "concept_confidence_score",
                    "mean",
                ),
                maximum_concept_confidence=(
                    "concept_confidence_score",
                    "max",
                ),
                mean_activation_fraction=(
                    "activation_fraction",
                    "mean",
                ),
                support_evidence_score=(
                    "support_score",
                    "sum",
                ),
                suppression_evidence_score=(
                    "suppression_score",
                    "sum",
                ),
            )
            .reset_index()
        )

        aggregates[
            "net_target_evidence_score"
        ] = (
            aggregates[
                "support_evidence_score"
            ]
            -
            aggregates[
                "suppression_evidence_score"
            ]
        )

        aggregates[
            "target_contradiction"
        ] = (
            aggregates[
                "supporting_concepts"
            ].gt(0)
            &
            aggregates[
                "suppressing_concepts"
            ].gt(0)
        )

        aggregates[
            "evidence_agreement_ratio"
        ] = (
            aggregates[
                [
                    "supporting_concepts",
                    "suppressing_concepts",
                ]
            ].max(axis=1)
            /
            aggregates[
                "active_concepts"
            ].replace(0, np.nan)
        )

    result = full_grid.merge(
        aggregates,
        on=[
            "game_pk",
            "team",
            "target",
        ],
        how="left",
        validate="one_to_one",
    )

    integer_columns = [
        "active_concepts",
        "active_strong_concepts",
        "active_candidate_concepts",
        "supporting_concepts",
        "suppressing_concepts",
        "unique_domains",
        "unique_concept_names",
    ]

    for column in integer_columns:
        if column not in result.columns:
            result[column] = 0

        result[column] = (
            result[column]
            .fillna(0)
            .astype("int64")
        )

    float_columns = [
        "support_evidence_score",
        "suppression_evidence_score",
        "net_target_evidence_score",
    ]

    for column in float_columns:
        if column not in result.columns:
            result[column] = 0.0

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    if "target_contradiction" not in result.columns:
        result["target_contradiction"] = False

    result["target_contradiction"] = (
        result["target_contradiction"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    result["prediction_created"] = False
    result["prediction_weight_assigned"] = False
    result["current_game_outcome_used"] = False
    result["future_games_used"] = False
    result["validated_out_of_sample"] = False
    result["requires_2025_validation"] = True
    result["target_state_engine_version"] = (
        ENGINE_VERSION
    )

    return result.sort_values(
        [
            "game_date",
            "game_pk",
            "target",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_target_state_summary(
    only_team: str | None = None,
) -> dict[str, Any]:
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Missing checkpoint directory: {CHECKPOINT_DIR}"
        )

    discovered_teams = sorted(
        path.name.replace(
            "_state_summary.parquet",
            "",
        )
        for path in CHECKPOINT_DIR.glob(
            "*_state_summary.parquet"
        )
    )

    if only_team is not None:
        alias_map = {
            "ARI": "AZ",
            "OAK": "ATH",
        }

        team = alias_map.get(
            str(only_team).upper(),
            str(only_team).upper(),
        )

        if team not in discovered_teams:
            raise ValueError(
                f"No state checkpoint found for team: {team}"
            )

        target_teams = [team]
    else:
        target_teams = discovered_teams

    frames = []

    for team in target_teams:
        team_states = (
            _build_team_target_states(team)
        )

        frames.append(team_states)

        print(
            f"Completed {team:<4} | "
            f"rows={len(team_states):,}"
        )

    target_states = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_rows = sum(
        len(_load_team_games(team))
        * len(TARGETS)
        for team in target_teams
    )

    duplicate_rows = int(
        target_states.duplicated(
            subset=[
                "game_pk",
                "team",
                "target",
            ]
        ).sum()
    )

    if len(target_states) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} target-state rows; "
            f"found {len(target_states):,}."
        )

    if duplicate_rows:
        raise AssertionError(
            f"Found {duplicate_rows:,} duplicate "
            "game-team-target rows."
        )

    full_run_complete = (
        len(target_teams) == 30
    )

    if full_run_complete:
        _atomic_parquet_write(
            target_states,
            TARGET_STATE_PATH,
        )

    metadata = {
        "engine": (
            "ATLAS Target-Specific "
            "Baseball State Summary Engine"
        ),
        "engine_version":
            ENGINE_VERSION,
        "learning_season":
            LEARNING_SEASON,
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "teams":
            int(len(target_teams)),
        "targets":
            int(len(TARGETS)),
        "target_state_rows":
            int(len(target_states)),
        "expected_rows":
            int(expected_rows),
        "duplicate_rows":
            duplicate_rows,
        "full_run_complete":
            full_run_complete,
        "output_path": (
            str(TARGET_STATE_PATH)
            if full_run_complete
            else None
        ),
        "policy": {
            "targets_kept_separate":
                True,
            "cross_target_netting_allowed":
                False,
            "current_game_outcome_used":
                False,
            "future_games_used":
                False,
            "prediction_created":
                False,
            "prediction_weights_assigned":
                False,
            "2025_validation_required":
                True,
        },
    }

    _atomic_json_write(
        metadata,
        METADATA_PATH,
    )

    print("=" * 78)
    print(
        "TARGET-SPECIFIC BASEBALL STATE SUMMARY"
    )
    print("=" * 78)
    print(
        f"Teams..................... "
        f"{len(target_teams):,}"
    )
    print(
        f"Targets................... "
        f"{len(TARGETS):,}"
    )
    print(
        f"Target-State Rows......... "
        f"{len(target_states):,}"
    )
    print(
        f"Duplicate Rows............ "
        f"{duplicate_rows:,}"
    )
    print(
        f"Full Master Built......... "
        f"{full_run_complete}"
    )
    print("=" * 78)

    return metadata
