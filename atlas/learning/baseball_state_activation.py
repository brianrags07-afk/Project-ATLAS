
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
LEARNING_SEASON = 2024

INTERACTION_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
)

CONCEPT_REGISTRY_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(LEARNING_SEASON)
    / "team_concept_registry.parquet"
)

CONCEPT_MEMBER_MAP_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(LEARNING_SEASON)
    / "team_concept_member_map.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "baseball_states"
    / str(LEARNING_SEASON)
)

TEAM_CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "team_checkpoints"
)

ACTIVE_CONCEPT_PATH = (
    OUTPUT_DIR
    / "team_game_active_concepts.parquet"
)

TEAM_GAME_STATE_PATH = (
    OUTPUT_DIR
    / "team_game_state_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "baseball_state_activation_metadata.json"
)


ALLOWED_CONCEPT_STATUSES = {
    "strong_concept_candidate",
    "concept_candidate",
}


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


def _load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_parquet(path)


def _normalize_interactions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="raise",
    ).dt.normalize()

    dataframe = dataframe[
        dataframe["atlas_season"].eq(
            LEARNING_SEASON
        )
    ].copy()

    dataframe = dataframe.sort_values(
        [
            "team",
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return dataframe


def _select_concepts(
    registry: pd.DataFrame,
    member_map: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    registry = registry[
        registry[
            "concept_lifecycle_status"
        ].isin(
            ALLOWED_CONCEPT_STATUSES
        )
    ].copy()

    selected_ids = set(
        registry[
            "concept_id"
        ].astype(str)
    )

    member_map = member_map[
        member_map[
            "concept_id"
        ].astype(str).isin(
            selected_ids
        )
    ].copy()

    if registry.empty:
        raise ValueError(
            "No eligible concept candidates found."
        )

    if member_map.empty:
        raise ValueError(
            "No concept members found for eligible concepts."
        )

    return registry, member_map


def _condition_is_active(
    values: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if operator == "<=":
        return numeric.le(threshold)

    if operator == ">=":
        return numeric.ge(threshold)

    if operator == "<":
        return numeric.lt(threshold)

    if operator == ">":
        return numeric.gt(threshold)

    if operator == "==":
        return numeric.eq(threshold)

    raise ValueError(
        f"Unsupported threshold operator: {operator}"
    )


def _required_active_members(
    total_members: int,
) -> int:
    if total_members <= 1:
        return 1

    if total_members == 2:
        return 1

    return max(
        2,
        int(
            np.ceil(
                total_members * 0.50
            )
        ),
    )


def _team_checkpoint_paths(
    team: str,
) -> tuple[Path, Path]:
    return (
        TEAM_CHECKPOINT_DIR
        / f"{team}_active_concepts.parquet",
        TEAM_CHECKPOINT_DIR
        / f"{team}_state_summary.parquet",
    )


def _team_checkpoint_complete(
    team: str,
) -> bool:
    active_path, summary_path = (
        _team_checkpoint_paths(team)
    )

    return (
        active_path.exists()
        and summary_path.exists()
    )


def _activate_one_team(
    team_rows: pd.DataFrame,
    team_registry: pd.DataFrame,
    team_members: pd.DataFrame,
    team: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    active_records: list[
        dict[str, Any]
    ] = []

    concept_lookup = (
        team_registry.set_index(
            "concept_id",
            drop=False,
        )
    )

    for concept_id, members in (
        team_members.groupby(
            "concept_id",
            sort=True,
        )
    ):
        concept_id = str(
            concept_id
        )

        if concept_id not in concept_lookup.index:
            continue

        concept = concept_lookup.loc[
            concept_id
        ]

        member_flags = []
        available_members = 0

        for member in members.itertuples(
            index=False
        ):
            feature = str(
                member.feature
            )

            if feature not in team_rows.columns:
                continue

            available_members += 1

            active = _condition_is_active(
                values=team_rows[feature],
                operator=str(
                    member.threshold_operator
                ),
                threshold=float(
                    member.threshold_value
                ),
            )

            member_flags.append(
                active.fillna(False)
            )

        if not member_flags:
            continue

        flag_matrix = pd.concat(
            member_flags,
            axis=1,
        )

        active_count = flag_matrix.sum(
            axis=1
        ).astype("int16")

        required_count = (
            _required_active_members(
                available_members
            )
        )

        concept_active = (
            active_count
            >= required_count
        )

        active_indices = (
            team_rows.index[
                concept_active
            ]
        )

        for row_index in active_indices:
            game_row = team_rows.loc[
                row_index
            ]

            count = int(
                active_count.loc[
                    row_index
                ]
            )

            active_records.append({
                "game_pk":
                    int(game_row["game_pk"]),
                "game_date":
                    game_row["game_date"],
                "atlas_season":
                    int(
                        game_row[
                            "atlas_season"
                        ]
                    ),
                "team":
                    str(game_row["team"]),
                "opponent":
                    str(
                        game_row.get(
                            "opponent",
                            "",
                        )
                    ),
                "home_away":
                    str(
                        game_row.get(
                            "home_away",
                            "",
                        )
                    ),
                "concept_id":
                    concept_id,
                "target":
                    str(concept["target"]),
                "concept_domain":
                    str(
                        concept[
                            "concept_domain"
                        ]
                    ),
                "concept_scope":
                    str(
                        concept[
                            "concept_scope"
                        ]
                    ),
                "batting_order_slot": (
                    None
                    if pd.isna(
                        concept[
                            "batting_order_slot"
                        ]
                    )
                    else str(
                        concept[
                            "batting_order_slot"
                        ]
                    )
                ),
                "concept_name":
                    str(
                        concept[
                            "concept_name"
                        ]
                    ),
                "effect_direction":
                    str(
                        concept[
                            "effect_direction"
                        ]
                    ),
                "concept_lifecycle_status":
                    str(
                        concept[
                            "concept_lifecycle_status"
                        ]
                    ),
                "concept_confidence_score":
                    float(
                        concept[
                            "concept_confidence_score"
                        ]
                    ),
                "weighted_lift":
                    float(
                        concept[
                            "weighted_lift"
                        ]
                    ),
                "total_concept_members":
                    int(
                        concept[
                            "member_count"
                        ]
                    ),
                "available_members":
                    int(
                        available_members
                    ),
                "active_members":
                    count,
                "required_active_members":
                    int(required_count),
                "activation_fraction":
                    float(
                        count
                        / available_members
                    ),
                "prediction_weight_assigned":
                    False,
                "validated_out_of_sample":
                    False,
                "requires_2025_validation":
                    True,
                "state_engine_version":
                    ENGINE_VERSION,
            })

    active = pd.DataFrame(
        active_records
    )

    if active.empty:
        active = pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "atlas_season",
                "team",
                "opponent",
                "home_away",
                "concept_id",
                "target",
                "concept_domain",
                "concept_scope",
                "batting_order_slot",
                "concept_name",
                "effect_direction",
                "concept_lifecycle_status",
                "concept_confidence_score",
                "weighted_lift",
                "total_concept_members",
                "available_members",
                "active_members",
                "required_active_members",
                "activation_fraction",
                "prediction_weight_assigned",
                "validated_out_of_sample",
                "requires_2025_validation",
                "state_engine_version",
            ]
        )

    if not active.empty:
        active = active.sort_values(
            [
                "game_date",
                "game_pk",
                "target",
                "concept_confidence_score",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    base_rows = team_rows[
        [
            column
            for column in [
                "game_pk",
                "game_date",
                "atlas_season",
                "team",
                "opponent",
                "home_away",
                "home_team",
                "away_team",
                "opposing_starting_pitcher_id",
                "complete_snapshot_join",
                "strict_backtest_safe",
            ]
            if column in team_rows.columns
        ]
    ].copy()

    if active.empty:
        summary = base_rows.copy()

        summary[
            "active_concepts"
        ] = 0

        summary[
            "active_strong_concepts"
        ] = 0

        summary[
            "active_supporting_concepts"
        ] = 0

        summary[
            "active_suppressing_concepts"
        ] = 0

        summary[
            "mean_active_confidence"
        ] = np.nan

        summary[
            "net_weighted_lift"
        ] = 0.0

        summary[
            "state_contradiction_count"
        ] = 0

    else:
        active = active.copy()

        active[
            "signed_confidence_lift"
        ] = (
            active[
                "concept_confidence_score"
            ]
            * active[
                "weighted_lift"
            ]
        )

        aggregates = (
            active.groupby(
                [
                    "game_pk",
                    "team",
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
                        (
                            values
                            == "strong_concept_candidate"
                        ).sum()
                    ),
                ),
                active_supporting_concepts=(
                    "effect_direction",
                    lambda values: int(
                        (
                            values
                            == "supports_target"
                        ).sum()
                    ),
                ),
                active_suppressing_concepts=(
                    "effect_direction",
                    lambda values: int(
                        (
                            values
                            == "suppresses_target"
                        ).sum()
                    ),
                ),
                mean_active_confidence=(
                    "concept_confidence_score",
                    "mean",
                ),
                net_weighted_lift=(
                    "signed_confidence_lift",
                    "sum",
                ),
            )
            .reset_index()
        )

        contradiction = (
            active.groupby(
                [
                    "game_pk",
                    "team",
                    "target",
                ],
                sort=False,
            )[
                "effect_direction"
            ]
            .nunique()
            .gt(1)
            .groupby(
                level=[
                    0,
                    1,
                ]
            )
            .sum()
            .rename(
                "state_contradiction_count"
            )
            .reset_index()
        )

        summary = base_rows.merge(
            aggregates,
            on=[
                "game_pk",
                "team",
            ],
            how="left",
            validate="one_to_one",
        )

        summary = summary.merge(
            contradiction,
            on=[
                "game_pk",
                "team",
            ],
            how="left",
            validate="one_to_one",
        )

        count_columns = [
            "active_concepts",
            "active_strong_concepts",
            "active_supporting_concepts",
            "active_suppressing_concepts",
            "state_contradiction_count",
        ]

        for column in count_columns:
            summary[column] = (
                summary[column]
                .fillna(0)
                .astype("int64")
            )

        summary[
            "net_weighted_lift"
        ] = (
            summary[
                "net_weighted_lift"
            ].fillna(0.0)
        )

    summary[
        "prediction_created"
    ] = False

    summary[
        "prediction_weight_assigned"
    ] = False

    summary[
        "current_game_outcome_used"
    ] = False

    summary[
        "future_games_used"
    ] = False

    summary[
        "requires_2025_validation"
    ] = True

    summary[
        "state_engine_version"
    ] = ENGINE_VERSION

    summary = summary.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return active, summary


def _assemble_master(
    teams: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    active_frames = []
    summary_frames = []

    for team in teams:
        active_path, summary_path = (
            _team_checkpoint_paths(team)
        )

        if active_path.exists():
            active_frames.append(
                pd.read_parquet(
                    active_path
                )
            )

        if summary_path.exists():
            summary_frames.append(
                pd.read_parquet(
                    summary_path
                )
            )

    active = (
        pd.concat(
            active_frames,
            ignore_index=True,
        )
        if active_frames
        else pd.DataFrame()
    )

    summary = (
        pd.concat(
            summary_frames,
            ignore_index=True,
        )
        if summary_frames
        else pd.DataFrame()
    )

    if not active.empty:
        active = active.sort_values(
            [
                "game_date",
                "game_pk",
                "team",
                "target",
                "concept_confidence_score",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    if not summary.empty:
        summary = summary.sort_values(
            [
                "game_date",
                "game_pk",
                "team",
            ],
            kind="stable",
        ).reset_index(drop=True)

    return active, summary


def run_baseball_state_activation(
    only_team: str | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    started = time.time()

    interactions = _normalize_interactions(
        _load_parquet(
            INTERACTION_PATH,
            "lineup-starter interaction table",
        )
    )

    concept_registry, member_map = (
        _select_concepts(
            registry=_load_parquet(
                CONCEPT_REGISTRY_PATH,
                "team concept registry",
            ),
            member_map=_load_parquet(
                CONCEPT_MEMBER_MAP_PATH,
                "concept member map",
            ),
        )
    )

    all_teams = sorted(
        interactions[
            "team"
        ].dropna().astype(str).unique()
    )

    if only_team is not None:
        requested = str(
            only_team
        ).upper()

        alias_map = {
            "ARI": "AZ",
            "OAK": "ATH",
        }

        requested = alias_map.get(
            requested,
            requested,
        )

        if requested not in all_teams:
            raise ValueError(
                f"Unknown ATLAS team code: {requested}"
            )

        target_teams = [
            requested
        ]

    elif limit is not None:
        target_teams = all_teams[
            :limit
        ]

    else:
        target_teams = all_teams

    TEAM_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete = [
        team
        for team in target_teams
        if (
            resume
            and _team_checkpoint_complete(
                team
            )
        )
    ]

    remaining = [
        team
        for team in target_teams
        if team not in complete
    ]

    print("=" * 78)
    print(
        "ATLAS BASEBALL STATE ACTIVATION ENGINE"
    )
    print("=" * 78)
    print(
        f"Season.................... "
        f"{LEARNING_SEASON}"
    )
    print(
        f"Team-Game Rows............ "
        f"{len(interactions):,}"
    )
    print(
        f"Eligible Concepts......... "
        f"{len(concept_registry):,}"
    )
    print(
        f"Underlying Members........ "
        f"{len(member_map):,}"
    )
    print(
        f"Target Teams.............. "
        f"{len(target_teams):,}"
    )
    print(
        f"Already Complete.......... "
        f"{len(complete):,}"
    )
    print(
        f"Remaining................. "
        f"{len(remaining):,}"
    )
    print(
        f"Checkpoint Directory...... "
        f"{TEAM_CHECKPOINT_DIR}"
    )
    print("=" * 78)

    newly_built = 0

    for index, team in enumerate(
        remaining,
        start=1,
    ):
        team_started = time.time()

        team_rows = interactions[
            interactions[
                "team"
            ].eq(team)
        ].copy()

        team_registry = concept_registry[
            concept_registry[
                "team"
            ].eq(team)
        ].copy()

        team_concept_ids = set(
            team_registry[
                "concept_id"
            ].astype(str)
        )

        team_members = member_map[
            member_map[
                "concept_id"
            ].astype(str).isin(
                team_concept_ids
            )
        ].copy()

        active, summary = (
            _activate_one_team(
                team_rows=team_rows,
                team_registry=
                    team_registry,
                team_members=
                    team_members,
                team=team,
            )
        )

        active_path, summary_path = (
            _team_checkpoint_paths(team)
        )

        _atomic_parquet_write(
            active,
            active_path,
        )

        _atomic_parquet_write(
            summary,
            summary_path,
        )

        newly_built += 1

        elapsed = (
            time.time()
            - team_started
        )

        print(
            f"Completed {team:<4} | "
            f"{len(complete) + newly_built:>2}/"
            f"{len(target_teams):<2} | "
            f"active rows={len(active):>7,} | "
            f"time={elapsed:>6.1f}s"
        )

    active, summary = (
        _assemble_master(
            target_teams
        )
    )

    complete_teams = int(
        summary[
            "team"
        ].nunique()
        if not summary.empty
        else 0
    )

    full_run_complete = bool(
        only_team is None
        and limit is None
        and complete_teams
        == len(all_teams)
    )

    if full_run_complete:
        _atomic_parquet_write(
            active,
            ACTIVE_CONCEPT_PATH,
        )

        _atomic_parquet_write(
            summary,
            TEAM_GAME_STATE_PATH,
        )

    duplicate_active = int(
        active.duplicated(
            subset=[
                "game_pk",
                "team",
                "concept_id",
            ]
        ).sum()
        if not active.empty
        else 0
    )

    duplicate_team_games = int(
        summary.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
        if not summary.empty
        else 0
    )

    if duplicate_active:
        raise AssertionError(
            f"Duplicate active concept rows: "
            f"{duplicate_active}"
        )

    if duplicate_team_games:
        raise AssertionError(
            f"Duplicate team-game state rows: "
            f"{duplicate_team_games}"
        )

    elapsed = time.time() - started

    result = {
        "engine":
            "ATLAS Baseball State Activation Engine",
        "engine_version":
            ENGINE_VERSION,
        "learning_season":
            LEARNING_SEASON,
        "all_teams":
            int(len(all_teams)),
        "target_teams":
            int(len(target_teams)),
        "teams_complete":
            complete_teams,
        "newly_built":
            int(newly_built),
        "eligible_concepts":
            int(len(concept_registry)),
        "concept_members":
            int(len(member_map)),
        "active_concept_rows":
            int(len(active)),
        "team_game_state_rows":
            int(len(summary)),
        "duplicate_active_rows":
            duplicate_active,
        "duplicate_team_game_rows":
            duplicate_team_games,
        "full_run_complete":
            full_run_complete,
        "elapsed_seconds":
            float(elapsed),
        "outputs": {
            "active_concepts": (
                str(ACTIVE_CONCEPT_PATH)
                if full_run_complete
                else None
            ),
            "team_game_states": (
                str(TEAM_GAME_STATE_PATH)
                if full_run_complete
                else None
            ),
            "checkpoint_directory":
                str(TEAM_CHECKPOINT_DIR),
        },
        "pregame_safety": {
            "current_game_outcome_used":
                False,
            "future_games_used":
                False,
            "predictions_created":
                False,
            "prediction_weights_assigned":
                False,
            "2025_validation_required":
                True,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("\n" + "=" * 78)
    print(
        "BASEBALL STATE ACTIVATION COMPLETE"
    )
    print("=" * 78)
    print(
        f"Teams Complete............ "
        f"{complete_teams:,}/"
        f"{len(target_teams):,}"
    )
    print(
        f"Active Concept Rows....... "
        f"{len(active):,}"
    )
    print(
        f"Team-Game State Rows...... "
        f"{len(summary):,}"
    )
    print(
        f"Duplicate Active Rows..... "
        f"{duplicate_active:,}"
    )
    print(
        f"Duplicate Team-Games...... "
        f"{duplicate_team_games:,}"
    )
    print(
        f"Full Master Built......... "
        f"{full_run_complete}"
    )
    print(
        f"Elapsed................... "
        f"{elapsed / 60:.1f} minutes"
    )
    print("=" * 78)

    return result
