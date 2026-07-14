
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
BACKTEST_SEASON = 2025

INTERACTION_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
)

TEAM_TARGET_PATH = (
    DATA_DIR
    / "backtest"
    / "targets"
    / "team_game_targets.parquet"
)

MEMBER_MAP_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / "2024"
    / "team_concept_member_map.parquet"
)

INTEGRATED_BELIEF_PATH = (
    DATA_DIR
    / "learning"
    / "integrated_beliefs"
    / "integrated_concept_belief_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "backtest"
    / "weighted_states"
    / str(BACKTEST_SEASON)
)

ACTIVE_PATH = (
    OUTPUT_DIR
    / "weighted_active_concepts.parquet"
)

STATE_PATH = (
    OUTPUT_DIR
    / "weighted_target_states.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "weighted_state_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "weighted_state_metadata.json"
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
    destination.parent.mkdir(parents=True, exist_ok=True)

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
    destination.parent.mkdir(parents=True, exist_ok=True)

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


def _normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="raise",
    ).dt.normalize()

    return dataframe


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
    if total_members <= 2:
        return 1

    return max(
        2,
        int(np.ceil(total_members * 0.50)),
    )


def _prepare_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    interactions = _normalize_dates(
        _load_parquet(
            INTERACTION_PATH,
            "lineup-starter interaction inputs",
        )
    )

    targets = _normalize_dates(
        _load_parquet(
            TEAM_TARGET_PATH,
            "team-game targets",
        )
    )

    beliefs = _load_parquet(
        INTEGRATED_BELIEF_PATH,
        "integrated concept beliefs",
    )

    members = _load_parquet(
        MEMBER_MAP_PATH,
        "concept member map",
    )

    interactions = interactions[
        interactions["atlas_season"].eq(
            BACKTEST_SEASON
        )
    ].copy()

    targets = targets[
        targets["atlas_season"].eq(
            BACKTEST_SEASON
        )
    ].copy()

    beliefs = beliefs[
        beliefs[
            "integrated_prediction_weight_ready"
        ].eq(True)
    ].copy()

    selected_ids = set(
        beliefs["concept_id"].astype(str)
    )

    members = members[
        members["concept_id"]
        .astype(str)
        .isin(selected_ids)
    ].copy()

    return interactions, targets, beliefs, members


def _activate_weighted_concepts(
    interactions: pd.DataFrame,
    beliefs: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for team in sorted(
        interactions["team"]
        .dropna()
        .astype(str)
        .unique()
    ):
        team_rows = interactions[
            interactions["team"].eq(team)
        ].copy()

        team_beliefs = beliefs[
            beliefs["team"].eq(team)
        ].copy()

        if team_beliefs.empty:
            continue

        belief_lookup = team_beliefs.set_index(
            "concept_id",
            drop=False,
        )

        concept_ids = set(
            team_beliefs["concept_id"]
            .astype(str)
        )

        team_members = members[
            members["concept_id"]
            .astype(str)
            .isin(concept_ids)
        ].copy()

        for concept_id, concept_members in (
            team_members.groupby(
                "concept_id",
                sort=True,
            )
        ):
            concept_id = str(concept_id)

            if concept_id not in belief_lookup.index:
                continue

            concept = belief_lookup.loc[concept_id]

            flags = []
            available_members = 0

            for member in concept_members.itertuples(
                index=False
            ):
                feature = str(member.feature)

                if feature not in team_rows.columns:
                    continue

                available_members += 1

                flags.append(
                    _condition_is_active(
                        team_rows[feature],
                        str(member.threshold_operator),
                        float(member.threshold_value),
                    ).fillna(False)
                )

            if not flags:
                continue

            flag_matrix = pd.concat(
                flags,
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

            active_mask = (
                active_count >= required_count
            )

            active_indices = team_rows.index[
                active_mask
            ]

            signed_weight = float(
                concept[
                    "integrated_signed_prediction_weight"
                ]
            )

            absolute_weight = abs(signed_weight)

            for row_index in active_indices:
                row = team_rows.loc[row_index]

                records.append({
                    "game_pk":
                        int(row["game_pk"]),
                    "game_date":
                        row["game_date"],
                    "atlas_season":
                        int(row["atlas_season"]),
                    "team":
                        str(row["team"]),
                    "opponent":
                        str(row.get("opponent", "")),
                    "home_away":
                        str(row.get("home_away", "")),
                    "concept_id":
                        concept_id,
                    "target":
                        str(concept["target"]),
                    "effect_direction":
                        str(
                            concept[
                                "effect_direction"
                            ]
                        ),
                    "validation_status":
                        str(
                            concept[
                                "validation_status"
                            ]
                        ),
                    "league_relationship":
                        str(
                            concept[
                                "league_relationship"
                            ]
                        ),
                    "integrated_belief_score":
                        float(
                            concept[
                                "league_adjusted_belief_score"
                            ]
                        ),
                    "signed_weight":
                        signed_weight,
                    "absolute_weight":
                        absolute_weight,
                    "active_members":
                        int(
                            active_count.loc[
                                row_index
                            ]
                        ),
                    "available_members":
                        int(available_members),
                    "required_active_members":
                        int(required_count),
                    "activation_fraction":
                        float(
                            active_count.loc[row_index]
                            / available_members
                        ),
                    "2026_outcomes_used":
                        False,
                    "engine_version":
                        ENGINE_VERSION,
                })

    active = pd.DataFrame(records)

    if not active.empty:
        active = active.sort_values(
            [
                "game_date",
                "game_pk",
                "team",
                "target",
                "absolute_weight",
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

    return active


def _build_target_states(
    interactions: pd.DataFrame,
    targets: pd.DataFrame,
    active: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
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
        if column in interactions.columns
    ]

    base = interactions[
        base_columns
    ].copy()

    target_frame = pd.DataFrame({
        "target": TARGETS,
    })

    base["_join_key"] = 1
    target_frame["_join_key"] = 1

    grid = base.merge(
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

        active["support_weight"] = np.where(
            active["signed_weight"].gt(0),
            active["absolute_weight"],
            0.0,
        )

        active["suppression_weight"] = np.where(
            active["signed_weight"].lt(0),
            active["absolute_weight"],
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
                support_concepts=(
                    "support_weight",
                    lambda values: int(
                        values.gt(0).sum()
                    ),
                ),
                suppression_concepts=(
                    "suppression_weight",
                    lambda values: int(
                        values.gt(0).sum()
                    ),
                ),
                support_weight=(
                    "support_weight",
                    "sum",
                ),
                suppression_weight=(
                    "suppression_weight",
                    "sum",
                ),
                net_weighted_state_score=(
                    "signed_weight",
                    "sum",
                ),
                total_absolute_weight=(
                    "absolute_weight",
                    "sum",
                ),
                maximum_concept_weight=(
                    "absolute_weight",
                    "max",
                ),
                mean_concept_belief=(
                    "integrated_belief_score",
                    "mean",
                ),
                mean_activation_fraction=(
                    "activation_fraction",
                    "mean",
                ),
            )
            .reset_index()
        )

    state = grid.merge(
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
        "support_concepts",
        "suppression_concepts",
    ]

    for column in integer_columns:
        state[column] = (
            state[column]
            .fillna(0)
            .astype("int64")
        )

    float_columns = [
        "support_weight",
        "suppression_weight",
        "net_weighted_state_score",
        "total_absolute_weight",
        "maximum_concept_weight",
    ]

    for column in float_columns:
        state[column] = (
            pd.to_numeric(
                state[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    state["state_contradiction"] = (
        state["support_concepts"].gt(0)
        & state["suppression_concepts"].gt(0)
    )

    state["state_has_evidence"] = (
        state["active_concepts"].gt(0)
    )

    target_long = targets.melt(
        id_vars=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
        ],
        value_vars=[
            target
            for target in TARGETS
            if target in targets.columns
        ],
        var_name="target",
        value_name="actual_outcome",
    )

    state = state.merge(
        target_long,
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
            "target",
        ],
        how="left",
        validate="one_to_one",
    )

    state["actual_outcome"] = pd.to_numeric(
        state["actual_outcome"],
        errors="coerce",
    )

    state["prediction_created"] = False
    state["probability_calibrated"] = False
    state["2026_outcomes_used"] = False
    state["engine_version"] = ENGINE_VERSION

    return state.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
            "target",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _build_summary(
    state: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        state.groupby(
            "target",
            sort=True,
        )
        .agg(
            rows=(
                "game_pk",
                "size",
            ),
            games_with_evidence=(
                "state_has_evidence",
                "sum",
            ),
            outcome_base_rate=(
                "actual_outcome",
                "mean",
            ),
            mean_active_concepts=(
                "active_concepts",
                "mean",
            ),
            mean_absolute_weight=(
                "total_absolute_weight",
                "mean",
            ),
            mean_net_state_score=(
                "net_weighted_state_score",
                "mean",
            ),
            contradiction_rows=(
                "state_contradiction",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["evidence_coverage"] = (
        summary["games_with_evidence"]
        / summary["rows"]
    )

    summary["backtest_season"] = (
        BACKTEST_SEASON
    )

    summary["engine_version"] = (
        ENGINE_VERSION
    )

    return summary


def run_weighted_state_backtest() -> dict[str, Any]:
    (
        interactions,
        targets,
        beliefs,
        members,
    ) = _prepare_inputs()

    active = _activate_weighted_concepts(
        interactions=interactions,
        beliefs=beliefs,
        members=members,
    )

    state = _build_target_states(
        interactions=interactions,
        targets=targets,
        active=active,
    )

    summary = _build_summary(state)

    expected_rows = (
        len(interactions)
        * len(TARGETS)
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

    duplicate_states = int(
        state.duplicated(
            subset=[
                "game_pk",
                "team",
                "target",
            ]
        ).sum()
    )

    if len(state) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} states; "
            f"found {len(state):,}."
        )

    if duplicate_active:
        raise AssertionError(
            f"Duplicate active concepts: {duplicate_active}"
        )

    if duplicate_states:
        raise AssertionError(
            f"Duplicate target states: {duplicate_states}"
        )

    _atomic_parquet_write(
        active,
        ACTIVE_PATH,
    )

    _atomic_parquet_write(
        state,
        STATE_PATH,
    )

    _atomic_parquet_write(
        summary,
        SUMMARY_PATH,
    )

    result = {
        "engine":
            "ATLAS 2025 Weighted State Backtest Engine",
        "engine_version":
            ENGINE_VERSION,
        "backtest_season":
            BACKTEST_SEASON,
        "team_game_rows":
            int(len(interactions)),
        "ready_concepts":
            int(len(beliefs)),
        "active_concept_rows":
            int(len(active)),
        "target_state_rows":
            int(len(state)),
        "expected_target_state_rows":
            int(expected_rows),
        "games_with_any_evidence":
            int(
                state[
                    "state_has_evidence"
                ].sum()
            ),
        "duplicate_active_rows":
            duplicate_active,
        "duplicate_target_states":
            duplicate_states,
        "probabilities_created":
            False,
        "2026_outcomes_used":
            False,
        "outputs": {
            "active_concepts":
                str(ACTIVE_PATH),
            "weighted_target_states":
                str(STATE_PATH),
            "summary":
                str(SUMMARY_PATH),
        },
        "policy": {
            "2024_concepts_frozen":
                True,
            "2025_used_for_combination_backtest":
                True,
            "2026_used":
                False,
            "probability_calibration_pending":
                True,
            "sportsbook_market_used":
                False,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS 2025 WEIGHTED STATE BACKTEST")
    print("=" * 78)
    print(
        f"2025 Team-Game Rows........ "
        f"{len(interactions):,}"
    )
    print(
        f"Ready Concepts............. "
        f"{len(beliefs):,}"
    )
    print(
        f"Active Concept Rows........ "
        f"{len(active):,}"
    )
    print(
        f"Target-State Rows.......... "
        f"{len(state):,}"
    )
    print(
        f"Expected Target States..... "
        f"{expected_rows:,}"
    )
    print(
        f"Duplicate Active Rows...... "
        f"{duplicate_active:,}"
    )
    print(
        f"Duplicate Target States.... "
        f"{duplicate_states:,}"
    )
    print(
        f"Probabilities Created...... False"
    )
    print(
        f"2026 Outcomes Used......... False"
    )
    print(
        f"Saved To................... "
        f"{STATE_PATH}"
    )
    print("=" * 78)

    return result
