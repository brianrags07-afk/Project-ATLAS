
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
LEARNING_SEASON = 2024

INTERACTION_INPUT_PATH = (
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

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "team_evidence"
    / "2024"
)

EVIDENCE_REGISTRY_PATH = (
    OUTPUT_DIR
    / "team_evidence_registry.parquet"
)

TEAM_SUMMARY_PATH = (
    OUTPUT_DIR
    / "team_evidence_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "team_evidence_metadata.json"
)


TARGET_COLUMNS = [
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


NON_FEATURE_COLUMNS = {
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
    "home_team",
    "away_team",
    "starting_lineup_ids",
    "starting_lineup_size",
    "starting_lineup_complete",
    "opposing_starting_pitcher_id",
    "source",
    "pregame_information_class",
    "lineup_engine_version",
    "interaction_engine_version",
    "strict_backtest_safe",
    "current_game_outcomes_used",
    "same_date_games_used",
    "future_games_used",
    "prediction_or_weight_assigned",
    "complete_snapshot_join",
    "starter_snapshot_matched",
    "batter_snapshot_slots_matched",
}


LEAKAGE_NAME_FRAGMENTS = {
    "runs_scored",
    "runs_allowed",
    "run_differential",
    "winner",
    "won",
    "lost",
    "home_win",
    "away_win",
    "game_total_runs",
    "final_score",
    "actual_result",
    "target_",
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


def _feature_family(
    feature: str,
) -> str:
    if feature.startswith("lineup_"):
        return "lineup_composition"

    if feature.startswith("starter_"):
        return "opposing_starter"

    if feature.startswith("slot_"):
        pieces = feature.split("_", 2)

        if len(pieces) >= 2:
            return f"batting_order_slot_{pieces[1]}"

        return "individual_batter_slot"

    if "home_away" in feature:
        return "game_context"

    return "other_pregame_evidence"


def _evidence_id(
    team: str,
    target: str,
    feature: str,
    direction: str,
) -> str:
    raw = (
        f"{team}|{target}|{feature}|"
        f"{direction}|{LEARNING_SEASON}"
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{team}_{target}_{direction}_"
        f"{digest}"
    ).upper()


def _numeric_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    candidates = []

    for column in dataframe.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        # Outcome labels must never be screened as pregame evidence.
        if column in TARGET_COLUMNS:
            continue

        lowered = column.lower()

        if any(
            fragment in lowered
            for fragment in LEAKAGE_NAME_FRAGMENTS
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            continue

        candidates.append(column)

    return sorted(candidates)


def _safe_rate(
    values: pd.Series,
) -> float | None:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def _safe_corr(
    feature: pd.Series,
    target: pd.Series,
) -> float | None:
    paired = pd.DataFrame({
        "feature": pd.to_numeric(
            feature,
            errors="coerce",
        ),
        "target": pd.to_numeric(
            target,
            errors="coerce",
        ),
    }).dropna()

    if len(paired) < 3:
        return None

    if (
        paired["feature"].nunique() < 2
        or paired["target"].nunique() < 2
    ):
        return None

    value = paired["feature"].corr(
        paired["target"]
    )

    if pd.isna(value):
        return None

    return float(value)


def _chronological_half_lifts(
    dataframe: pd.DataFrame,
    feature: str,
    target: str,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, float | None]:
    ordered = dataframe.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    midpoint = len(ordered) // 2

    halves = {
        "first_half": ordered.iloc[:midpoint],
        "second_half": ordered.iloc[midpoint:],
    }

    output: dict[str, float | None] = {}

    for name, half in halves.items():
        baseline = _safe_rate(
            half[target]
        )

        low = half[
            pd.to_numeric(
                half[feature],
                errors="coerce",
            ).le(lower_threshold)
        ]

        high = half[
            pd.to_numeric(
                half[feature],
                errors="coerce",
            ).ge(upper_threshold)
        ]

        low_rate = _safe_rate(
            low[target]
        )

        high_rate = _safe_rate(
            high[target]
        )

        output[
            f"{name}_low_lift"
        ] = (
            low_rate - baseline
            if (
                low_rate is not None
                and baseline is not None
            )
            else None
        )

        output[
            f"{name}_high_lift"
        ] = (
            high_rate - baseline
            if (
                high_rate is not None
                and baseline is not None
            )
            else None
        )

    return output


def _direction_consistent(
    lift_a: float | None,
    lift_b: float | None,
) -> bool:
    if lift_a is None or lift_b is None:
        return False

    if lift_a == 0 or lift_b == 0:
        return False

    return bool(
        np.sign(lift_a)
        == np.sign(lift_b)
    )


def _confidence_score(
    sample_size: int,
    absolute_lift: float,
    direction_consistent: bool,
    coverage: float,
    correlation: float | None,
) -> float:
    sample_component = min(
        sample_size / 60.0,
        1.0,
    )

    lift_component = min(
        absolute_lift / 0.20,
        1.0,
    )

    consistency_component = (
        1.0
        if direction_consistent
        else 0.0
    )

    coverage_component = min(
        max(coverage, 0.0),
        1.0,
    )

    correlation_component = min(
        abs(correlation or 0.0) / 0.30,
        1.0,
    )

    score = (
        0.30 * sample_component
        + 0.30 * lift_component
        + 0.20 * consistency_component
        + 0.10 * coverage_component
        + 0.10 * correlation_component
    )

    return float(
        min(max(score, 0.0), 1.0)
    )


def _candidate_status(
    sample_size: int,
    absolute_lift: float,
    direction_consistent: bool,
    confidence: float,
) -> str:
    if (
        sample_size >= 30
        and absolute_lift >= 0.15
        and direction_consistent
        and confidence >= 0.70
    ):
        return "strong_candidate"

    if (
        sample_size >= 20
        and absolute_lift >= 0.10
        and direction_consistent
        and confidence >= 0.50
    ):
        return "candidate"

    if (
        sample_size >= 15
        and absolute_lift >= 0.07
        and confidence >= 0.35
    ):
        return "weak_candidate"

    return "insufficient_sample"


def discover_team_target_evidence(
    team_df: pd.DataFrame,
    team: str,
    target: str,
    feature_columns: list[str],
    minimum_bucket_sample: int = 10,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    target_values = pd.to_numeric(
        team_df[target],
        errors="coerce",
    )

    baseline_rate = _safe_rate(
        target_values
    )

    if (
        baseline_rate is None
        or target_values.nunique(
            dropna=True
        ) < 2
    ):
        return records

    total_games = int(
        len(team_df)
    )

    for feature in feature_columns:
        if feature == target or feature in TARGET_COLUMNS:
            continue

        feature_values = pd.to_numeric(
            team_df[feature],
            errors="coerce",
        )

        valid_mask = (
            feature_values.notna()
            & target_values.notna()
        )

        valid = team_df.loc[
            valid_mask,
            [
                "game_pk",
                "game_date",
                feature,
                target,
            ],
        ].copy()

        if len(valid) < (
            minimum_bucket_sample * 2
        ):
            continue

        if valid[feature].nunique() < 5:
            continue

        lower_threshold = float(
            valid[feature].quantile(0.25)
        )

        upper_threshold = float(
            valid[feature].quantile(0.75)
        )

        if (
            not np.isfinite(lower_threshold)
            or not np.isfinite(upper_threshold)
            or lower_threshold
            >= upper_threshold
        ):
            continue

        low_group = valid[
            valid[feature]
            <= lower_threshold
        ]

        high_group = valid[
            valid[feature]
            >= upper_threshold
        ]

        if (
            len(low_group)
            < minimum_bucket_sample
            or len(high_group)
            < minimum_bucket_sample
        ):
            continue

        low_rate = _safe_rate(
            low_group[target]
        )

        high_rate = _safe_rate(
            high_group[target]
        )

        if (
            low_rate is None
            or high_rate is None
        ):
            continue

        low_lift = float(
            low_rate - baseline_rate
        )

        high_lift = float(
            high_rate - baseline_rate
        )

        correlation = _safe_corr(
            valid[feature],
            valid[target],
        )

        half_lifts = (
            _chronological_half_lifts(
                dataframe=valid,
                feature=feature,
                target=target,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
        )

        candidates = [
            {
                "direction": "low",
                "threshold_operator": "<=",
                "threshold_value":
                    lower_threshold,
                "sample_size": int(
                    len(low_group)
                ),
                "observed_rate": float(
                    low_rate
                ),
                "lift": low_lift,
                "first_half_lift":
                    half_lifts[
                        "first_half_low_lift"
                    ],
                "second_half_lift":
                    half_lifts[
                        "second_half_low_lift"
                    ],
            },
            {
                "direction": "high",
                "threshold_operator": ">=",
                "threshold_value":
                    upper_threshold,
                "sample_size": int(
                    len(high_group)
                ),
                "observed_rate": float(
                    high_rate
                ),
                "lift": high_lift,
                "first_half_lift":
                    half_lifts[
                        "first_half_high_lift"
                    ],
                "second_half_lift":
                    half_lifts[
                        "second_half_high_lift"
                    ],
            },
        ]

        for candidate in candidates:
            absolute_lift = abs(
                candidate["lift"]
            )

            consistent = (
                _direction_consistent(
                    candidate[
                        "first_half_lift"
                    ],
                    candidate[
                        "second_half_lift"
                    ],
                )
            )

            coverage = (
                candidate["sample_size"]
                / total_games
                if total_games
                else 0.0
            )

            confidence = (
                _confidence_score(
                    sample_size=candidate[
                        "sample_size"
                    ],
                    absolute_lift=absolute_lift,
                    direction_consistent=consistent,
                    coverage=coverage,
                    correlation=correlation,
                )
            )

            status = _candidate_status(
                sample_size=candidate[
                    "sample_size"
                ],
                absolute_lift=absolute_lift,
                direction_consistent=consistent,
                confidence=confidence,
            )

            if status == "insufficient_sample":
                continue

            records.append({
                "evidence_id": _evidence_id(
                    team=team,
                    target=target,
                    feature=feature,
                    direction=candidate[
                        "direction"
                    ],
                ),
                "learning_season":
                    LEARNING_SEASON,
                "team": team,
                "target": target,
                "feature": feature,
                "feature_family":
                    _feature_family(feature),
                "direction": candidate[
                    "direction"
                ],
                "threshold_operator":
                    candidate[
                        "threshold_operator"
                    ],
                "threshold_value":
                    candidate[
                        "threshold_value"
                    ],
                "team_games_observed":
                    total_games,
                "feature_rows_available":
                    int(len(valid)),
                "coverage": float(
                    len(valid) / total_games
                ),
                "condition_sample_size":
                    candidate[
                        "sample_size"
                    ],
                "target_base_rate":
                    float(baseline_rate),
                "condition_observed_rate":
                    candidate[
                        "observed_rate"
                    ],
                "lift": candidate["lift"],
                "absolute_lift":
                    absolute_lift,
                "relative_lift": (
                    candidate["lift"]
                    / baseline_rate
                    if baseline_rate > 0
                    else None
                ),
                "feature_target_correlation":
                    correlation,
                "first_half_lift":
                    candidate[
                        "first_half_lift"
                    ],
                "second_half_lift":
                    candidate[
                        "second_half_lift"
                    ],
                "chronological_direction_consistent":
                    consistent,
                "confidence_score":
                    confidence,
                "lifecycle_status":
                    status,
                "predictive_weight_assigned":
                    False,
                "validated_out_of_sample":
                    False,
                "single_metric_control_allowed":
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

    return records


def build_team_evidence_registry(
    interaction_inputs: pd.DataFrame,
    team_targets: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    inputs = interaction_inputs.copy()
    targets = team_targets.copy()

    inputs["game_date"] = pd.to_datetime(
        inputs["game_date"],
        errors="coerce",
    ).dt.normalize()

    targets["game_date"] = pd.to_datetime(
        targets["game_date"],
        errors="coerce",
    ).dt.normalize()

    target_columns = [
        column
        for column in TARGET_COLUMNS
        if column in targets.columns
    ]

    if not target_columns:
        raise KeyError(
            "No configured discovery targets "
            "were found."
        )

    target_join_columns = [
        "game_pk",
        "team",
        "atlas_season",
        "game_date",
    ] + target_columns

    combined = inputs.merge(
        targets[
            target_join_columns
        ],
        on=[
            "game_pk",
            "team",
            "atlas_season",
            "game_date",
        ],
        how="inner",
        validate="one_to_one",
    )

    learning = combined[
        combined["atlas_season"]
        .eq(LEARNING_SEASON)
    ].copy()

    if learning.empty:
        raise ValueError(
            f"No rows found for {LEARNING_SEASON}."
        )

    feature_columns = (
        _numeric_feature_columns(
            learning
        )
    )

    records: list[dict[str, Any]] = []
    summary_records: list[
        dict[str, Any]
    ] = []

    teams = sorted(
        str(value)
        for value in learning[
            "team"
        ].dropna().unique()
    )

    for team in teams:
        team_df = learning[
            learning["team"].eq(team)
        ].copy()

        for target in target_columns:
            target_records = (
                discover_team_target_evidence(
                    team_df=team_df,
                    team=team,
                    target=target,
                    feature_columns=feature_columns,
                )
            )

            records.extend(
                target_records
            )

            target_base_rate = (
                _safe_rate(
                    team_df[target]
                )
            )

            summary_records.append({
                "learning_season":
                    LEARNING_SEASON,
                "team": team,
                "target": target,
                "games": int(
                    len(team_df)
                ),
                "target_base_rate":
                    target_base_rate,
                "features_screened": int(
                    len(feature_columns)
                ),
                "evidence_objects_found":
                    int(
                        len(target_records)
                    ),
                "strong_candidates": int(
                    sum(
                        record[
                            "lifecycle_status"
                        ]
                        == "strong_candidate"
                        for record
                        in target_records
                    )
                ),
                "candidates": int(
                    sum(
                        record[
                            "lifecycle_status"
                        ]
                        == "candidate"
                        for record
                        in target_records
                    )
                ),
                "weak_candidates": int(
                    sum(
                        record[
                            "lifecycle_status"
                        ]
                        == "weak_candidate"
                        for record
                        in target_records
                    )
                ),
                "validated_out_of_sample":
                    False,
                "requires_2025_validation":
                    True,
                "engine_version":
                    ENGINE_VERSION,
            })

    registry = pd.DataFrame(
        records
    )

    summary = pd.DataFrame(
        summary_records
    )

    if not registry.empty:
        registry = registry.sort_values(
            [
                "team",
                "target",
                "confidence_score",
                "absolute_lift",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    summary = summary.sort_values(
        [
            "team",
            "target",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return registry, summary


def validate_team_evidence(
    registry: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    expected_teams = 30

    teams = int(
        summary["team"].nunique()
    )

    if teams != expected_teams:
        raise AssertionError(
            f"Expected {expected_teams} teams; "
            f"found {teams}."
        )

    duplicate_ids = int(
        registry[
            "evidence_id"
        ].duplicated().sum()
        if not registry.empty
        else 0
    )

    if duplicate_ids:
        raise AssertionError(
            f"Found {duplicate_ids} duplicate "
            "evidence IDs."
        )

    weights_assigned = int(
        registry[
            "predictive_weight_assigned"
        ].fillna(False).sum()
        if not registry.empty
        else 0
    )

    if weights_assigned:
        raise AssertionError(
            "Discovery engine assigned predictive "
            "weights unexpectedly."
        )

    return {
        "learning_season":
            LEARNING_SEASON,
        "teams": teams,
        "targets": int(
            summary[
                "target"
            ].nunique()
        ),
        "team_target_rows": int(
            len(summary)
        ),
        "evidence_objects": int(
            len(registry)
        ),
        "strong_candidates": int(
            (
                registry[
                    "lifecycle_status"
                ]
                == "strong_candidate"
            ).sum()
            if not registry.empty
            else 0
        ),
        "candidates": int(
            (
                registry[
                    "lifecycle_status"
                ]
                == "candidate"
            ).sum()
            if not registry.empty
            else 0
        ),
        "weak_candidates": int(
            (
                registry[
                    "lifecycle_status"
                ]
                == "weak_candidate"
            ).sum()
            if not registry.empty
            else 0
        ),
        "duplicate_evidence_ids":
            duplicate_ids,
        "predictive_weights_assigned":
            weights_assigned,
    }


def run_team_evidence_discovery() -> dict[str, Any]:
    interaction_inputs = _load_parquet(
        INTERACTION_INPUT_PATH,
        "lineup starter interaction inputs",
    )

    team_targets = _load_parquet(
        TEAM_TARGET_PATH,
        "team game targets",
    )

    registry, summary = (
        build_team_evidence_registry(
            interaction_inputs=
                interaction_inputs,
            team_targets=team_targets,
        )
    )

    validation = validate_team_evidence(
        registry=registry,
        summary=summary,
    )

    _atomic_parquet_write(
        registry,
        EVIDENCE_REGISTRY_PATH,
    )

    _atomic_parquet_write(
        summary,
        TEAM_SUMMARY_PATH,
    )

    metadata = {
        "engine": (
            "ATLAS Team Evidence "
            "Discovery Engine"
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
        "validation": validation,
        "outputs": {
            "evidence_registry": str(
                EVIDENCE_REGISTRY_PATH
            ),
            "team_summary": str(
                TEAM_SUMMARY_PATH
            ),
        },
        "learning_policy": {
            "team_local_learning": True,
            "league_wide_weights_used":
                False,
            "prediction_weights_assigned":
                False,
            "single_metric_control_allowed":
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

    print("=" * 78)
    print(
        "ATLAS TEAM EVIDENCE "
        "DISCOVERY ENGINE"
    )
    print("=" * 78)
    print(
        f"Learning Season........... "
        f"{validation['learning_season']}"
    )
    print(
        f"Teams Learned............. "
        f"{validation['teams']:,}"
    )
    print(
        f"Targets per Team.......... "
        f"{validation['targets']:,}"
    )
    print(
        f"Team-Target Rows.......... "
        f"{validation['team_target_rows']:,}"
    )
    print(
        f"Evidence Objects.......... "
        f"{validation['evidence_objects']:,}"
    )
    print(
        f"Strong Candidates......... "
        f"{validation['strong_candidates']:,}"
    )
    print(
        f"Candidates................ "
        f"{validation['candidates']:,}"
    )
    print(
        f"Weak Candidates........... "
        f"{validation['weak_candidates']:,}"
    )
    print(
        f"Duplicate Evidence IDs.... "
        f"{validation['duplicate_evidence_ids']:,}"
    )
    print(
        f"Prediction Weights........ "
        f"{validation['predictive_weights_assigned']:,}"
    )
    print(
        f"Saved To.................. "
        f"{EVIDENCE_REGISTRY_PATH}"
    )
    print("=" * 78)

    return metadata
