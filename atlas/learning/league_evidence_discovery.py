
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR
from atlas.learning.team_evidence_discovery import (
    ENGINE_VERSION as DISCOVERY_ENGINE_VERSION,
    _numeric_feature_columns,
    discover_team_target_evidence,
)


ENGINE_VERSION = "1.1.0"
LEARNING_SEASON = 2024

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

GAME_TARGET_PATH = (
    DATA_DIR
    / "backtest"
    / "targets"
    / "game_targets.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "league_evidence"
    / str(LEARNING_SEASON)
)

TEAM_EVIDENCE_PATH = (
    OUTPUT_DIR
    / "league_team_evidence_registry.parquet"
)

GAME_EVIDENCE_PATH = (
    OUTPUT_DIR
    / "league_game_environment_registry.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "league_evidence_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "league_evidence_metadata.json"
)


TEAM_TARGETS = [
    "won",
    "lost",
    "team_scored_5_plus",
    "team_scored_3_or_less",
    "team_scored_8_plus",
    "team_allowed_3_or_less",
    "team_allowed_5_plus",
]

GAME_TARGETS = [
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


def _normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="raise",
    ).dt.normalize()

    return dataframe


def _prepare_team_learning_table(
    interactions: pd.DataFrame,
    team_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    available_targets = [
        target
        for target in TEAM_TARGETS
        if target in team_targets.columns
    ]

    keys = [
        "game_pk",
        "game_date",
        "atlas_season",
        "team",
    ]

    combined = interactions.merge(
        team_targets[
            keys + available_targets
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    learning = combined[
        combined["atlas_season"].eq(
            LEARNING_SEASON
        )
    ].copy()

    learning = learning.sort_values(
        ["game_date", "game_pk", "team"],
        kind="stable",
    ).reset_index(drop=True)

    features = _numeric_feature_columns(
        learning
    )

    leaked = sorted(
        set(features)
        & set(TEAM_TARGETS)
        & set(GAME_TARGETS)
    )

    if leaked:
        raise AssertionError(
            f"Target leakage detected: {leaked}"
        )

    return learning, features


def _paired_game_features(
    interactions: pd.DataFrame,
    game_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    season_rows = interactions[
        interactions["atlas_season"].eq(
            LEARNING_SEASON
        )
    ].copy()

    numeric_features = _numeric_feature_columns(
        season_rows
    )

    # Preserve both sides separately. This allows ATLAS to learn
    # interactions such as strong home lineup + vulnerable away
    # starter without blending the teams together.
    home = season_rows[
        season_rows["home_away"].eq("HOME")
    ][
        [
            "game_pk",
            "game_date",
            "atlas_season",
        ]
        + numeric_features
    ].copy()

    away = season_rows[
        season_rows["home_away"].eq("AWAY")
    ][
        [
            "game_pk",
            "game_date",
            "atlas_season",
        ]
        + numeric_features
    ].copy()

    home = home.rename(
        columns={
            column: f"home_{column}"
            for column in numeric_features
        }
    )

    away = away.rename(
        columns={
            column: f"away_{column}"
            for column in numeric_features
        }
    )

    paired = home.merge(
        away,
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
        ],
        how="inner",
        validate="one_to_one",
    )

    available_targets = [
        target
        for target in GAME_TARGETS
        if target in game_targets.columns
    ]

    paired = paired.merge(
        game_targets[
            [
                "game_pk",
                "game_date",
                "atlas_season",
            ]
            + available_targets
        ],
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
        ],
        how="inner",
        validate="one_to_one",
    )

    paired = paired.sort_values(
        ["game_date", "game_pk"],
        kind="stable",
    ).reset_index(drop=True)

    feature_columns = [
        column
        for column in paired.columns
        if (
            column.startswith("home_")
            or column.startswith("away_")
        )
        and column not in available_targets
        and pd.api.types.is_numeric_dtype(
            paired[column]
        )
    ]

    return paired, feature_columns


def _league_feature_family(
    feature: str,
) -> str:
    feature = str(feature)

    side = None

    if feature.startswith("home_"):
        side = "home"
        feature = feature[5:]

    elif feature.startswith("away_"):
        side = "away"
        feature = feature[5:]

    if feature.startswith("lineup_"):
        family = "lineup_composition"

    elif feature.startswith("starter_"):
        family = "opposing_starter"

    elif feature.startswith("slot_"):
        pieces = feature.split("_", 2)

        family = (
            f"batting_order_slot_{pieces[1]}"
            if len(pieces) > 1
            else "individual_batter_slot"
        )

    else:
        family = "other_pregame_evidence"

    return (
        f"{side}_{family}"
        if side is not None
        else family
    )


def _league_evidence_id(
    scope_name: str,
    target: str,
    feature: str,
    direction: str,
) -> str:
    raw = (
        f"{LEARNING_SEASON}|{scope_name}|"
        f"{target}|{feature}|{direction}"
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:14]

    return (
        f"{scope_name}_{target}_{direction}_{digest}"
    ).upper()


def _two_proportion_p_value(
    successes_a: int,
    sample_a: int,
    successes_b: int,
    sample_b: int,
) -> float | None:
    if sample_a <= 0 or sample_b <= 0:
        return None

    rate_a = successes_a / sample_a
    rate_b = successes_b / sample_b

    pooled = (
        (successes_a + successes_b)
        / (sample_a + sample_b)
    )

    variance = (
        pooled
        * (1.0 - pooled)
        * (
            (1.0 / sample_a)
            + (1.0 / sample_b)
        )
    )

    if variance <= 0:
        return None

    z_score = (
        (rate_a - rate_b)
        / math.sqrt(variance)
    )

    # Two-sided normal approximation.
    return float(
        math.erfc(
            abs(z_score)
            / math.sqrt(2.0)
        )
    )


def _benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        p_values,
        errors="coerce",
    )

    valid = numeric.dropna()

    output = pd.Series(
        np.nan,
        index=p_values.index,
        dtype="float64",
    )

    if valid.empty:
        return output

    ordered = valid.sort_values(
        kind="stable"
    )

    total = len(ordered)

    adjusted = (
        ordered
        * total
        / np.arange(
            1,
            total + 1,
            dtype="float64",
        )
    )

    # Enforce monotonic adjusted p-values.
    adjusted = (
        adjusted.iloc[::-1]
        .cummin()
        .iloc[::-1]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    output.loc[
        ordered.index
    ] = adjusted.values

    return output


def _chronological_league_lifts(
    dataframe: pd.DataFrame,
    feature: str,
    target: str,
    operator: str,
    threshold: float,
) -> tuple[
    float | None,
    float | None,
    bool,
]:
    ordered = dataframe.sort_values(
        ["game_date", "game_pk"],
        kind="stable",
    ).reset_index(drop=True)

    midpoint = len(ordered) // 2

    lifts = []

    for half in [
        ordered.iloc[:midpoint],
        ordered.iloc[midpoint:],
    ]:
        feature_values = pd.to_numeric(
            half[feature],
            errors="coerce",
        )

        target_values = pd.to_numeric(
            half[target],
            errors="coerce",
        )

        valid = (
            feature_values.notna()
            & target_values.notna()
        )

        feature_values = feature_values[valid]
        target_values = target_values[valid]

        if operator == "<=":
            condition = (
                feature_values <= threshold
            )
        else:
            condition = (
                feature_values >= threshold
            )

        if (
            condition.sum() == 0
            or (~condition).sum() == 0
        ):
            lifts.append(None)
            continue

        condition_rate = float(
            target_values[
                condition
            ].mean()
        )

        comparison_rate = float(
            target_values[
                ~condition
            ].mean()
        )

        lifts.append(
            condition_rate
            - comparison_rate
        )

    first_lift, second_lift = lifts

    consistent = bool(
        first_lift is not None
        and second_lift is not None
        and first_lift != 0
        and second_lift != 0
        and np.sign(first_lift)
        == np.sign(second_lift)
    )

    return (
        first_lift,
        second_lift,
        consistent,
    )


def _league_candidate_status(
    q_value: float | None,
    absolute_lift: float,
    sample_size: int,
    chronological_consistent: bool,
) -> str:
    if q_value is None or pd.isna(q_value):
        return "insufficient_evidence"

    if (
        q_value <= 0.01
        and absolute_lift >= 0.025
        and sample_size >= 200
        and chronological_consistent
    ):
        return "strong_candidate"

    if (
        q_value <= 0.05
        and absolute_lift >= 0.015
        and sample_size >= 150
    ):
        return "candidate"

    if (
        q_value <= 0.10
        and absolute_lift >= 0.010
        and sample_size >= 100
    ):
        return "weak_candidate"

    return "insufficient_evidence"


def discover_league_target_evidence(
    dataframe: pd.DataFrame,
    scope_name: str,
    target: str,
    feature_columns: list[str],
    minimum_condition_sample: int = 100,
) -> list[dict[str, Any]]:
    target_values = pd.to_numeric(
        dataframe[target],
        errors="coerce",
    )

    target_base_rate = float(
        target_values.mean()
    )

    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        if feature == target:
            continue

        feature_values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        valid_mask = (
            feature_values.notna()
            & target_values.notna()
        )

        valid_feature = (
            feature_values[
                valid_mask
            ]
        )

        valid_target = (
            target_values[
                valid_mask
            ]
        )

        if len(valid_feature) < (
            minimum_condition_sample * 3
        ):
            continue

        if valid_feature.nunique() < 5:
            continue

        lower_threshold = float(
            valid_feature.quantile(0.25)
        )

        upper_threshold = float(
            valid_feature.quantile(0.75)
        )

        if (
            not np.isfinite(lower_threshold)
            or not np.isfinite(upper_threshold)
            or lower_threshold >= upper_threshold
        ):
            continue

        candidates = [
            (
                "low",
                "<=",
                lower_threshold,
                valid_feature
                <= lower_threshold,
            ),
            (
                "high",
                ">=",
                upper_threshold,
                valid_feature
                >= upper_threshold,
            ),
        ]

        for (
            direction,
            operator,
            threshold,
            condition_mask,
        ) in candidates:
            comparison_mask = (
                ~condition_mask
            )

            condition_sample = int(
                condition_mask.sum()
            )

            comparison_sample = int(
                comparison_mask.sum()
            )

            if (
                condition_sample
                < minimum_condition_sample
                or comparison_sample
                < minimum_condition_sample
            ):
                continue

            condition_successes = int(
                valid_target[
                    condition_mask
                ].sum()
            )

            comparison_successes = int(
                valid_target[
                    comparison_mask
                ].sum()
            )

            condition_rate = (
                condition_successes
                / condition_sample
            )

            comparison_rate = (
                comparison_successes
                / comparison_sample
            )

            lift_vs_comparison = float(
                condition_rate
                - comparison_rate
            )

            p_value = (
                _two_proportion_p_value(
                    successes_a=
                        condition_successes,
                    sample_a=
                        condition_sample,
                    successes_b=
                        comparison_successes,
                    sample_b=
                        comparison_sample,
                )
            )

            (
                first_half_lift,
                second_half_lift,
                chronological_consistent,
            ) = _chronological_league_lifts(
                dataframe=dataframe.loc[
                    valid_mask,
                    [
                        "game_pk",
                        "game_date",
                        feature,
                        target,
                    ],
                ].copy(),
                feature=feature,
                target=target,
                operator=operator,
                threshold=threshold,
            )

            records.append({
                "evidence_id":
                    _league_evidence_id(
                        scope_name=scope_name,
                        target=target,
                        feature=feature,
                        direction=direction,
                    ),
                "learning_season":
                    LEARNING_SEASON,
                "team":
                    scope_name,
                "learning_scope":
                    scope_name,
                "target":
                    target,
                "feature":
                    feature,
                "feature_family":
                    _league_feature_family(
                        feature
                    ),
                "direction":
                    direction,
                "threshold_operator":
                    operator,
                "threshold_value":
                    threshold,
                "rows_observed":
                    int(len(dataframe)),
                "feature_rows_available":
                    int(len(valid_feature)),
                "coverage":
                    float(
                        len(valid_feature)
                        / len(dataframe)
                    ),
                "condition_sample_size":
                    condition_sample,
                "comparison_sample_size":
                    comparison_sample,
                "target_base_rate":
                    target_base_rate,
                "condition_observed_rate":
                    float(condition_rate),
                "comparison_observed_rate":
                    float(comparison_rate),
                "lift":
                    lift_vs_comparison,
                "absolute_lift":
                    abs(lift_vs_comparison),
                "relative_lift": (
                    lift_vs_comparison
                    / comparison_rate
                    if comparison_rate > 0
                    else None
                ),
                "first_half_lift":
                    first_half_lift,
                "second_half_lift":
                    second_half_lift,
                "chronological_direction_consistent":
                    chronological_consistent,
                "p_value":
                    p_value,
                "q_value":
                    None,
                "confidence_score":
                    None,
                "lifecycle_status":
                    None,
                "league_prior_only":
                    True,
                "team_local_override_allowed":
                    True,
                "predictive_weight_assigned":
                    False,
                "validated_out_of_sample":
                    False,
                "requires_2025_validation":
                    True,
                "created_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "engine_version":
                    ENGINE_VERSION,
            })

    if not records:
        return []

    result = pd.DataFrame(
        records
    )

    result["q_value"] = (
        _benjamini_hochberg(
            result["p_value"]
        )
    )

    result["lifecycle_status"] = [
        _league_candidate_status(
            q_value=row.q_value,
            absolute_lift=row.absolute_lift,
            sample_size=row.condition_sample_size,
            chronological_consistent=(
                row.chronological_direction_consistent
            ),
        )
        for row in result.itertuples(
            index=False
        )
    ]

    # League confidence reflects statistical reliability,
    # effect magnitude, sample size, and chronological agreement.
    significance_component = (
        1.0
        - result["q_value"]
        .fillna(1.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    effect_component = (
        result["absolute_lift"]
        / 0.05
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    sample_component = (
        result[
            "condition_sample_size"
        ]
        / 500.0
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    consistency_component = (
        result[
            "chronological_direction_consistent"
        ]
        .fillna(False)
        .astype(float)
    )

    result["confidence_score"] = (
        0.40 * significance_component
        + 0.25 * effect_component
        + 0.20 * sample_component
        + 0.15 * consistency_component
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    result = result[
        ~result[
            "lifecycle_status"
        ].eq(
            "insufficient_evidence"
        )
    ].copy()

    return result.to_dict(
        orient="records"
    )



def _discover_scope(
    dataframe: pd.DataFrame,
    scope_name: str,
    targets: list[str],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence_records: list[
        dict[str, Any]
    ] = []

    summary_records: list[
        dict[str, Any]
    ] = []

    print(
        f"\n[{scope_name}] "
        f"Rows={len(dataframe):,} | "
        f"Features={len(feature_columns):,}"
    )

    for index, target in enumerate(
        targets,
        start=1,
    ):
        if target not in dataframe.columns:
            continue

        records = (
            discover_league_target_evidence(
                dataframe=dataframe,
                scope_name=scope_name,
                target=target,
                feature_columns=feature_columns,
                minimum_condition_sample=100,
            )
        )

        for record in records:
            record[
                "learning_scope"
            ] = scope_name

            record[
                "league_prior_only"
            ] = True

            record[
                "team_local_override_allowed"
            ] = True

        evidence_records.extend(
            records
        )

        summary_records.append({
            "learning_season":
                LEARNING_SEASON,
            "learning_scope":
                scope_name,
            "target":
                target,
            "rows_observed":
                int(len(dataframe)),
            "features_screened":
                int(len(feature_columns)),
            "evidence_objects":
                int(len(records)),
            "strong_candidates":
                int(sum(
                    record[
                        "lifecycle_status"
                    ] == "strong_candidate"
                    for record in records
                )),
            "candidates":
                int(sum(
                    record[
                        "lifecycle_status"
                    ] == "candidate"
                    for record in records
                )),
            "weak_candidates":
                int(sum(
                    record[
                        "lifecycle_status"
                    ] == "weak_candidate"
                    for record in records
                )),
            "prediction_weights_assigned":
                False,
            "requires_2025_validation":
                True,
            "engine_version":
                ENGINE_VERSION,
        })

        print(
            f"  Target {index:>2}/{len(targets)} "
            f"{target:<28} "
            f"objects={len(records):>6,}"
        )

    evidence = pd.DataFrame(
        evidence_records
    )

    summary = pd.DataFrame(
        summary_records
    )

    if not evidence.empty:
        evidence = evidence.sort_values(
            [
                "target",
                "confidence_score",
                "absolute_lift",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    return evidence, summary


def validate_league_discovery(
    team_learning: pd.DataFrame,
    game_learning: pd.DataFrame,
    team_evidence: pd.DataFrame,
    game_evidence: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    expected_team_rows = int(
        team_learning[
            "game_pk"
        ].nunique()
        * 2
    )

    duplicate_game_rows = int(
        game_learning[
            "game_pk"
        ].duplicated().sum()
    )

    weights_assigned = int(
        summary[
            "prediction_weights_assigned"
        ].fillna(False).sum()
    )

    if len(team_learning) != expected_team_rows:
        raise AssertionError(
            f"Expected {expected_team_rows:,} "
            f"team-game rows; found "
            f"{len(team_learning):,}."
        )

    if duplicate_game_rows:
        raise AssertionError(
            f"Found {duplicate_game_rows} "
            "duplicate league game rows."
        )

    if weights_assigned:
        raise AssertionError(
            "League discovery assigned "
            "prediction weights."
        )

    return {
        "learning_season":
            LEARNING_SEASON,
        "team_game_learning_rows":
            int(len(team_learning)),
        "unique_games":
            int(
                game_learning[
                    "game_pk"
                ].nunique()
            ),
        "league_team_evidence_objects":
            int(len(team_evidence)),
        "league_game_evidence_objects":
            int(len(game_evidence)),
        "total_evidence_objects":
            int(
                len(team_evidence)
                + len(game_evidence)
            ),
        "summary_rows":
            int(len(summary)),
        "duplicate_game_rows":
            duplicate_game_rows,
        "prediction_weights_assigned":
            weights_assigned,
    }


def run_league_evidence_discovery() -> dict[str, Any]:
    interactions = _normalize_dates(
        _load_parquet(
            INTERACTION_PATH,
            "interaction inputs",
        )
    )

    team_targets = _normalize_dates(
        _load_parquet(
            TEAM_TARGET_PATH,
            "team-game targets",
        )
    )

    game_targets = _normalize_dates(
        _load_parquet(
            GAME_TARGET_PATH,
            "game targets",
        )
    )

    (
        team_learning,
        team_feature_columns,
    ) = _prepare_team_learning_table(
        interactions=interactions,
        team_targets=team_targets,
    )

    (
        game_learning,
        game_feature_columns,
    ) = _paired_game_features(
        interactions=interactions,
        game_targets=game_targets,
    )

    (
        team_evidence,
        team_summary,
    ) = _discover_scope(
        dataframe=team_learning,
        scope_name="MLB_TEAM_BASELINE",
        targets=TEAM_TARGETS,
        feature_columns=
            team_feature_columns,
    )

    (
        game_evidence,
        game_summary,
    ) = _discover_scope(
        dataframe=game_learning,
        scope_name="MLB_GAME_ENVIRONMENT",
        targets=GAME_TARGETS,
        feature_columns=
            game_feature_columns,
    )

    summary = pd.concat(
        [
            team_summary,
            game_summary,
        ],
        ignore_index=True,
    )

    validation = (
        validate_league_discovery(
            team_learning=team_learning,
            game_learning=game_learning,
            team_evidence=team_evidence,
            game_evidence=game_evidence,
            summary=summary,
        )
    )

    _atomic_parquet_write(
        team_evidence,
        TEAM_EVIDENCE_PATH,
    )

    _atomic_parquet_write(
        game_evidence,
        GAME_EVIDENCE_PATH,
    )

    _atomic_parquet_write(
        summary,
        SUMMARY_PATH,
    )

    metadata = {
        "engine": (
            "ATLAS League-Wide "
            "Evidence Discovery Engine"
        ),
        "engine_version":
            ENGINE_VERSION,
        "discovery_engine_version":
            DISCOVERY_ENGINE_VERSION,
        "learning_season":
            LEARNING_SEASON,
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "validation":
            validation,
        "outputs": {
            "league_team_registry": str(
                TEAM_EVIDENCE_PATH
            ),
            "league_game_registry": str(
                GAME_EVIDENCE_PATH
            ),
            "league_summary": str(
                SUMMARY_PATH
            ),
        },
        "learning_policy": {
            "league_prior_only": True,
            "team_local_override_allowed":
                True,
            "team_specific_learning_replaced":
                False,
            "prediction_weights_assigned":
                False,
            "2025_validation_required":
                True,
            "2026_used_for_learning":
                False,
        },
    }

    _atomic_json_write(
        metadata,
        METADATA_PATH,
    )

    print("\n" + "=" * 78)
    print(
        "ATLAS LEAGUE-WIDE EVIDENCE DISCOVERY"
    )
    print("=" * 78)
    print(
        f"Team-Game Learning Rows.... "
        f"{validation['team_game_learning_rows']:,}"
    )
    print(
        f"Unique Games............... "
        f"{validation['unique_games']:,}"
    )
    print(
        f"League Team Evidence....... "
        f"{validation['league_team_evidence_objects']:,}"
    )
    print(
        f"Game Environment Evidence.. "
        f"{validation['league_game_evidence_objects']:,}"
    )
    print(
        f"Total Evidence Objects..... "
        f"{validation['total_evidence_objects']:,}"
    )
    print(
        f"Prediction Weights......... "
        f"{validation['prediction_weights_assigned']:,}"
    )
    print("=" * 78)

    return metadata
