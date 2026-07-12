
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
PREDICTION_SEASON = 2026

INTERACTION_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
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

CALIBRATION_REGISTRY_PATH = (
    DATA_DIR
    / "calibration"
    / "probabilities"
    / "2025"
    / "target_probability_calibration_registry.parquet"
)

CALIBRATION_KNOTS_PATH = (
    DATA_DIR
    / "calibration"
    / "probabilities"
    / "2025"
    / "target_probability_calibration_knots.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "predictions"
    / str(PREDICTION_SEASON)
)

ACTIVE_CONCEPT_PATH = (
    OUTPUT_DIR
    / "pregame_active_concepts.parquet"
)

TARGET_PREDICTION_PATH = (
    OUTPUT_DIR
    / "pregame_target_predictions.parquet"
)

GAME_PREDICTION_PATH = (
    OUTPUT_DIR
    / "pregame_game_predictions.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "pregame_prediction_metadata.json"
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
            PREDICTION_SEASON
        )
    ].copy()

    return dataframe.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)


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
        int(
            np.ceil(
                total_members * 0.50
            )
        ),
    )


def _apply_calibration(
    score: float,
    knots: pd.DataFrame,
) -> float:
    ordered = knots.sort_values(
        "score_threshold",
        kind="stable",
    )

    x = pd.to_numeric(
        ordered["score_threshold"],
        errors="raise",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        ordered["calibrated_probability"],
        errors="raise",
    ).to_numpy(dtype=float)

    return float(
        np.interp(
            score,
            x,
            y,
            left=y[0],
            right=y[-1],
        )
    )


def _prepare_inputs(
    game_date: str | None,
    only_team: str | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    interactions = _normalize_interactions(
        _load_parquet(
            INTERACTION_PATH,
            "pregame interaction table",
        )
    )

    if game_date is not None:
        selected_date = pd.Timestamp(
            game_date
        ).normalize()

        interactions = interactions[
            interactions["game_date"].eq(
                selected_date
            )
        ].copy()

    if only_team is not None:
        alias_map = {
            "ARI": "AZ",
            "OAK": "ATH",
        }

        team = alias_map.get(
            str(only_team).upper(),
            str(only_team).upper(),
        )

        interactions = interactions[
            interactions["team"].eq(team)
        ].copy()

    if interactions.empty:
        raise ValueError(
            "No matching 2026 pregame rows were found."
        )

    beliefs = _load_parquet(
        INTEGRATED_BELIEF_PATH,
        "integrated belief registry",
    )

    beliefs = beliefs[
        beliefs[
            "integrated_prediction_weight_ready"
        ].eq(True)
    ].copy()

    members = _load_parquet(
        MEMBER_MAP_PATH,
        "concept member map",
    )

    selected_ids = set(
        beliefs["concept_id"]
        .astype(str)
    )

    members = members[
        members["concept_id"]
        .astype(str)
        .isin(selected_ids)
    ].copy()

    calibration_registry = _load_parquet(
        CALIBRATION_REGISTRY_PATH,
        "probability calibration registry",
    )

    accepted_calibrations = (
        calibration_registry[
            calibration_registry[
                "calibration_accepted"
            ].eq(True)
        ]
        .copy()
    )

    knots = _load_parquet(
        CALIBRATION_KNOTS_PATH,
        "probability calibration knots",
    )

    return (
        interactions,
        beliefs,
        members,
        accepted_calibrations,
        knots,
    )


def _activate_concepts(
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

        belief_lookup = (
            team_beliefs.set_index(
                "concept_id",
                drop=False,
            )
        )

        concept_ids = set(
            team_beliefs[
                "concept_id"
            ].astype(str)
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

            concept = belief_lookup.loc[
                concept_id
            ]

            flags = []
            available_members = 0

            for member in (
                concept_members.itertuples(
                    index=False
                )
            ):
                feature = str(member.feature)

                if feature not in team_rows.columns:
                    continue

                available_members += 1

                flags.append(
                    _condition_is_active(
                        values=
                            team_rows[feature],
                        operator=str(
                            member.threshold_operator
                        ),
                        threshold=float(
                            member.threshold_value
                        ),
                    ).fillna(False)
                )

            if not flags:
                continue

            flag_matrix = pd.concat(
                flags,
                axis=1,
            )

            active_count = (
                flag_matrix.sum(axis=1)
                .astype("int16")
            )

            required_count = (
                _required_active_members(
                    available_members
                )
            )

            active_indices = team_rows.index[
                active_count.ge(
                    required_count
                )
            ]

            signed_weight = float(
                concept[
                    "integrated_signed_prediction_weight"
                ]
            )

            for row_index in active_indices:
                row = team_rows.loc[
                    row_index
                ]

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
                        str(
                            row.get(
                                "opponent",
                                "",
                            )
                        ),
                    "home_away":
                        str(
                            row.get(
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
                        abs(signed_weight),
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
                            active_count.loc[
                                row_index
                            ]
                            / available_members
                        ),
                    "current_game_outcome_used":
                        False,
                    "future_games_used":
                        False,
                    "sportsbook_used":
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


def _build_target_predictions(
    interactions: pd.DataFrame,
    active: pd.DataFrame,
    calibration_registry: pd.DataFrame,
    knots: pd.DataFrame,
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
            )
            .reset_index()
        )

    prediction = grid.merge(
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
        prediction[column] = (
            prediction[column]
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
        prediction[column] = (
            pd.to_numeric(
                prediction[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    accepted_targets = set(
        calibration_registry[
            "target"
        ].astype(str)
    )

    calibration_lookup = {
        target: group.copy()
        for target, group in knots.groupby(
            "target",
            sort=False,
        )
    }

    probabilities = []
    statuses = []

    for row in prediction.itertuples(
        index=False
    ):
        target = str(row.target)
        active_concepts = int(
            row.active_concepts
        )

        if target not in accepted_targets:
            probabilities.append(np.nan)
            statuses.append(
                "calibration_unavailable"
            )
            continue

        if active_concepts <= 0:
            probabilities.append(np.nan)
            statuses.append(
                "no_active_evidence"
            )
            continue

        target_knots = calibration_lookup.get(
            target
        )

        if target_knots is None:
            probabilities.append(np.nan)
            statuses.append(
                "calibration_knots_missing"
            )
            continue

        probability = _apply_calibration(
            score=float(
                row.net_weighted_state_score
            ),
            knots=target_knots,
        )

        probabilities.append(
            probability
        )

        statuses.append(
            "calibrated_prediction"
        )

    prediction[
        "calibrated_probability"
    ] = probabilities

    prediction[
        "prediction_status"
    ] = statuses

    prediction[
        "prediction_ready"
    ] = (
        prediction[
            "prediction_status"
        ].eq(
            "calibrated_prediction"
        )
    )

    prediction[
        "state_contradiction"
    ] = (
        prediction[
            "support_concepts"
        ].gt(0)
        & prediction[
            "suppression_concepts"
        ].gt(0)
    )

    prediction[
        "evidence_strength"
    ] = np.select(
        [
            prediction[
                "active_concepts"
            ].ge(3),
            prediction[
                "active_concepts"
            ].eq(2),
            prediction[
                "active_concepts"
            ].eq(1),
        ],
        [
            "multi_concept",
            "two_concept",
            "single_concept",
        ],
        default="none",
    )

    prediction[
        "current_game_outcome_used"
    ] = False

    prediction["future_games_used"] = False
    prediction["sportsbook_used"] = False
    prediction["calibration_season"] = 2025
    prediction["prediction_engine_version"] = (
        ENGINE_VERSION
    )

    prediction["predicted_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    return prediction.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
            "target",
        ],
        kind="stable",
    ).reset_index(drop=True)



def _build_game_predictions(
    target_predictions: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = [
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
        ]
        if column in target_predictions.columns
    ]

    # Preserve every team-game, even when ATLAS abstains
    # on all targets for that row.
    base = (
        target_predictions[
            identity_columns
        ]
        .drop_duplicates(
            subset=[
                "game_pk",
                "team",
            ]
        )
        .copy()
    )

    probability_long = target_predictions[
        target_predictions[
            "calibrated_probability"
        ].notna()
    ][
        [
            "game_pk",
            "team",
            "target",
            "calibrated_probability",
        ]
    ].copy()

    if probability_long.empty:
        probability_wide = base[
            [
                "game_pk",
                "team",
            ]
        ].copy()

    else:
        probability_wide = (
            probability_long.pivot(
                index=[
                    "game_pk",
                    "team",
                ],
                columns="target",
                values="calibrated_probability",
            )
            .reset_index()
        )

        probability_wide.columns.name = None

        probability_wide = probability_wide.rename(
            columns={
                target: f"{target}_probability"
                for target in TARGETS
                if target in probability_wide.columns
            }
        )

    ready_counts = (
        target_predictions.groupby(
            [
                "game_pk",
                "team",
            ],
            sort=False,
        )
        .agg(
            calibrated_targets=(
                "prediction_ready",
                "sum",
            ),
            active_concept_instances=(
                "active_concepts",
                "sum",
            ),
            contradiction_targets=(
                "state_contradiction",
                "sum",
            ),
        )
        .reset_index()
    )

    game_predictions = (
        base.merge(
            probability_wide,
            on=[
                "game_pk",
                "team",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            ready_counts,
            on=[
                "game_pk",
                "team",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    integer_columns = [
        "calibrated_targets",
        "active_concept_instances",
        "contradiction_targets",
    ]

    for column in integer_columns:
        game_predictions[column] = (
            game_predictions[column]
            .fillna(0)
            .astype("int64")
        )

    game_predictions[
        "any_prediction_ready"
    ] = (
        game_predictions[
            "calibrated_targets"
        ].gt(0)
    )

    win_column = "won_probability"
    loss_column = "lost_probability"

    if (
        win_column in game_predictions.columns
        and loss_column in game_predictions.columns
    ):
        game_predictions[
            "win_loss_probability_sum"
        ] = (
            game_predictions[win_column]
            + game_predictions[loss_column]
        )

        game_predictions[
            "win_loss_both_available"
        ] = (
            game_predictions[win_column].notna()
            & game_predictions[loss_column].notna()
        )

        game_predictions[
            "win_loss_consistency_gap"
        ] = (
            game_predictions[
                "win_loss_probability_sum"
            ]
            - 1.0
        ).abs()

    else:
        game_predictions[
            "win_loss_probability_sum"
        ] = np.nan

        game_predictions[
            "win_loss_both_available"
        ] = False

        game_predictions[
            "win_loss_consistency_gap"
        ] = np.nan

    # --------------------------------------------------------
    # Reconcile independently calibrated win/loss signals.
    # At least two available signals are required.
    # --------------------------------------------------------
    probability_lookup = (
        game_predictions.set_index(
            ["game_pk", "team"],
            drop=False,
        )
    )

    raw_probabilities = []
    source_counts = []
    source_labels = []

    for row in game_predictions.itertuples(index=False):
        sources = []

        own_win = getattr(
            row,
            "won_probability",
            np.nan,
        )
        own_loss = getattr(
            row,
            "lost_probability",
            np.nan,
        )

        if pd.notna(own_win):
            sources.append(
                ("team_won", float(own_win))
            )

        if pd.notna(own_loss):
            sources.append(
                (
                    "inverse_team_lost",
                    1.0 - float(own_loss),
                )
            )

        opponent_key = (
            row.game_pk,
            row.opponent,
        )

        if opponent_key in probability_lookup.index:
            opponent_row = probability_lookup.loc[
                opponent_key
            ]

            opponent_win = opponent_row.get(
                "won_probability",
                np.nan,
            )

            opponent_loss = opponent_row.get(
                "lost_probability",
                np.nan,
            )

            if pd.notna(opponent_loss):
                sources.append(
                    (
                        "opponent_lost",
                        float(opponent_loss),
                    )
                )

            if pd.notna(opponent_win):
                sources.append(
                    (
                        "inverse_opponent_won",
                        1.0 - float(opponent_win),
                    )
                )

        source_counts.append(len(sources))
        source_labels.append(
            "|".join(
                label
                for label, _ in sources
            )
        )

        if len(sources) >= 2:
            raw_probabilities.append(
                float(
                    np.mean(
                        [
                            probability
                            for _, probability
                            in sources
                        ]
                    )
                )
            )
        else:
            raw_probabilities.append(np.nan)

    game_predictions[
        "moneyline_raw_win_probability"
    ] = raw_probabilities

    game_predictions[
        "moneyline_signal_count"
    ] = source_counts

    game_predictions[
        "moneyline_signal_sources"
    ] = source_labels

    game_predictions[
        "reconciled_win_probability"
    ] = np.nan

    game_predictions[
        "moneyline_prediction_status"
    ] = "insufficient_signals"

    for game_pk, indices in game_predictions.groupby(
        "game_pk",
        sort=False,
    ).groups.items():
        indices = list(indices)

        if len(indices) != 2:
            continue

        first_index, second_index = indices

        first_raw = game_predictions.at[
            first_index,
            "moneyline_raw_win_probability",
        ]

        second_raw = game_predictions.at[
            second_index,
            "moneyline_raw_win_probability",
        ]

        if pd.notna(first_raw) and pd.notna(second_raw):
            total = float(first_raw + second_raw)

            if total > 0:
                game_predictions.at[
                    first_index,
                    "reconciled_win_probability",
                ] = float(first_raw / total)

                game_predictions.at[
                    second_index,
                    "reconciled_win_probability",
                ] = float(second_raw / total)

                game_predictions.loc[
                    [first_index, second_index],
                    "moneyline_prediction_status",
                ] = "reconciled_both_teams"

        elif pd.notna(first_raw):
            game_predictions.at[
                first_index,
                "reconciled_win_probability",
            ] = float(first_raw)

            game_predictions.at[
                second_index,
                "reconciled_win_probability",
            ] = float(1.0 - first_raw)

            game_predictions.loc[
                [first_index, second_index],
                "moneyline_prediction_status",
            ] = "reconciled_one_team_complement"

        elif pd.notna(second_raw):
            game_predictions.at[
                second_index,
                "reconciled_win_probability",
            ] = float(second_raw)

            game_predictions.at[
                first_index,
                "reconciled_win_probability",
            ] = float(1.0 - second_raw)

            game_predictions.loc[
                [first_index, second_index],
                "moneyline_prediction_status",
            ] = "reconciled_one_team_complement"

    game_predictions[
        "moneyline_prediction_ready"
    ] = (
        game_predictions[
            "reconciled_win_probability"
        ].notna()
    )

    game_predictions[
        "moneyline_probability_method"
    ] = np.where(
        game_predictions[
            "moneyline_prediction_ready"
        ],
        "multi_signal_reconciliation",
        "abstain",
    )

    game_predictions["sportsbook_used"] = False
    game_predictions["current_game_outcome_used"] = False
    game_predictions["future_games_used"] = False
    game_predictions["prediction_engine_version"] = (
        ENGINE_VERSION
    )

    return game_predictions.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

def run_pregame_prediction_engine(
    game_date: str | None = None,
    only_team: str | None = None,
    save_master: bool = False,
) -> dict[str, Any]:
    (
        interactions,
        beliefs,
        members,
        calibration_registry,
        knots,
    ) = _prepare_inputs(
        game_date=game_date,
        only_team=only_team,
    )

    active = _activate_concepts(
        interactions=interactions,
        beliefs=beliefs,
        members=members,
    )

    target_predictions = (
        _build_target_predictions(
            interactions=interactions,
            active=active,
            calibration_registry=
                calibration_registry,
            knots=knots,
        )
    )

    game_predictions = (
        _build_game_predictions(
            target_predictions
        )
    )

    expected_target_rows = (
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

    duplicate_target_predictions = int(
        target_predictions.duplicated(
            subset=[
                "game_pk",
                "team",
                "target",
            ]
        ).sum()
    )

    if (
        len(target_predictions)
        != expected_target_rows
    ):
        raise AssertionError(
            f"Expected {expected_target_rows:,} "
            f"target predictions; found "
            f"{len(target_predictions):,}."
        )

    if duplicate_active:
        raise AssertionError(
            f"Duplicate active concepts: "
            f"{duplicate_active}"
        )

    if duplicate_target_predictions:
        raise AssertionError(
            f"Duplicate target predictions: "
            f"{duplicate_target_predictions}"
        )

    if target_predictions[
        "current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Current-game outcomes were used."
        )

    if target_predictions[
        "future_games_used"
    ].any():
        raise AssertionError(
            "Future games were used."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    date_label = (
        pd.Timestamp(game_date)
        .strftime("%Y%m%d")
        if game_date is not None
        else "all_available"
    )

    team_label = (
        str(only_team).upper()
        if only_team is not None
        else "all_teams"
    )

    run_dir = (
        OUTPUT_DIR
        / "runs"
        / f"{date_label}_{team_label}"
    )

    _atomic_parquet_write(
        active,
        run_dir
        / "active_concepts.parquet",
    )

    _atomic_parquet_write(
        target_predictions,
        run_dir
        / "target_predictions.parquet",
    )

    _atomic_parquet_write(
        game_predictions,
        run_dir
        / "game_predictions.parquet",
    )

    if save_master:
        _atomic_parquet_write(
            active,
            ACTIVE_CONCEPT_PATH,
        )

        _atomic_parquet_write(
            target_predictions,
            TARGET_PREDICTION_PATH,
        )

        _atomic_parquet_write(
            game_predictions,
            GAME_PREDICTION_PATH,
        )

    status_counts = (
        target_predictions[
            "prediction_status"
        ].value_counts()
    )

    result = {
        "engine":
            "ATLAS 2026 Pregame Prediction Engine",
        "engine_version":
            ENGINE_VERSION,
        "prediction_season":
            PREDICTION_SEASON,
        "selected_game_date":
            game_date,
        "selected_team":
            only_team,
        "team_game_rows":
            int(len(interactions)),
        "unique_games":
            int(
                interactions[
                    "game_pk"
                ].nunique()
            ),
        "ready_concepts_available":
            int(len(beliefs)),
        "active_concept_rows":
            int(len(active)),
        "target_prediction_rows":
            int(
                len(target_predictions)
            ),
        "game_prediction_rows":
            int(len(game_predictions)),
        "calibrated_predictions":
            int(
                status_counts.get(
                    "calibrated_prediction",
                    0,
                )
            ),
        "no_active_evidence":
            int(
                status_counts.get(
                    "no_active_evidence",
                    0,
                )
            ),
        "calibration_unavailable":
            int(
                status_counts.get(
                    "calibration_unavailable",
                    0,
                )
            ),
        "duplicate_active_rows":
            duplicate_active,
        "duplicate_target_predictions":
            duplicate_target_predictions,
        "sportsbook_used":
            False,
        "current_game_outcomes_used":
            False,
        "future_games_used":
            False,
        "master_outputs_saved":
            bool(save_master),
        "outputs": {
            "run_directory":
                str(run_dir),
            "active_concepts":
                str(
                    run_dir
                    / "active_concepts.parquet"
                ),
            "target_predictions":
                str(
                    run_dir
                    / "target_predictions.parquet"
                ),
            "game_predictions":
                str(
                    run_dir
                    / "game_predictions.parquet"
                ),
        },
        "policy": {
            "2024_discovery_frozen":
                True,
            "2025_validation_used":
                True,
            "2025_calibration_used":
                True,
            "2026_outcomes_used":
                False,
            "market_used":
                False,
            "uncalibrated_targets_abstain":
                True,
            "no_evidence_targets_abstain":
                True,
        },
    }

    _atomic_json_write(
        result,
        run_dir
        / "prediction_metadata.json",
    )

    if save_master:
        _atomic_json_write(
            result,
            METADATA_PATH,
        )

    print("=" * 78)
    print(
        "ATLAS 2026 PREGAME PREDICTION ENGINE"
    )
    print("=" * 78)
    print(
        f"Team-Game Rows............ "
        f"{len(interactions):,}"
    )
    print(
        f"Unique Games.............. "
        f"{interactions['game_pk'].nunique():,}"
    )
    print(
        f"Ready Concepts Available.. "
        f"{len(beliefs):,}"
    )
    print(
        f"Active Concept Rows....... "
        f"{len(active):,}"
    )
    print(
        f"Target Prediction Rows.... "
        f"{len(target_predictions):,}"
    )
    print(
        f"Calibrated Predictions.... "
        f"{result['calibrated_predictions']:,}"
    )
    print(
        f"No Active Evidence........ "
        f"{result['no_active_evidence']:,}"
    )
    print(
        f"Calibration Unavailable... "
        f"{result['calibration_unavailable']:,}"
    )
    print(
        f"Duplicate Active Rows..... "
        f"{duplicate_active:,}"
    )
    print(
        f"Duplicate Target Rows..... "
        f"{duplicate_target_predictions:,}"
    )
    print(
        "2026 Outcomes Used........ False"
    )
    print(
        "Sportsbook Used........... False"
    )
    print(
        f"Saved To.................. "
        f"{run_dir}"
    )
    print("=" * 78)

    return result
