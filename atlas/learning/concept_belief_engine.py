
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
DISCOVERY_SEASON = 2024
VALIDATION_SEASON = 2025

CONCEPT_REGISTRY_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(DISCOVERY_SEASON)
    / "team_concept_registry.parquet"
)

VALIDATION_REGISTRY_PATH = (
    DATA_DIR
    / "validation"
    / "concepts"
    / str(VALIDATION_SEASON)
    / "concept_validation_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "concept_beliefs"
)

BELIEF_REGISTRY_PATH = (
    OUTPUT_DIR
    / "concept_belief_registry.parquet"
)

BELIEF_SUMMARY_PATH = (
    OUTPUT_DIR
    / "concept_belief_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "concept_belief_metadata.json"
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


def _clip01(
    values: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )


def _discovery_status_component(
    status: pd.Series,
) -> pd.Series:
    mapping = {
        "strong_concept_candidate": 1.00,
        "concept_candidate": 0.72,
        "weak_concept_candidate": 0.38,
    }

    return (
        status.astype(str)
        .map(mapping)
        .fillna(0.25)
        .astype(float)
    )


def _validation_status_component(
    status: pd.Series,
) -> pd.Series:
    mapping = {
        "validated_strong": 1.00,
        "validated": 0.90,
        "direction_retained_weak": 0.66,
        "not_confirmed": 0.36,
        "reversed": 0.12,
        "reversed_strong": 0.02,
        "insufficient_2025_sample": 0.45,
    }

    return (
        status.astype(str)
        .map(mapping)
        .fillna(0.35)
        .astype(float)
    )


def _belief_grade(
    score: float,
) -> str:
    if score >= 0.88:
        return "elite"

    if score >= 0.76:
        return "strong"

    if score >= 0.64:
        return "reliable"

    if score >= 0.50:
        return "possible"

    if score >= 0.35:
        return "weak"

    return "ignore"


def _lifecycle_stage(
    validation_status: str,
    belief_score: float,
) -> str:
    if validation_status in {
        "reversed",
        "reversed_strong",
    }:
        return "identity_change_risk"

    if validation_status == "insufficient_2025_sample":
        return "sample_pending"

    if belief_score >= 0.76:
        return "trusted_2026_candidate"

    if belief_score >= 0.50:
        return "monitor_2026"

    return "research_only"


def _prediction_weight(
    belief_score: pd.Series,
    validation_status: pd.Series,
) -> pd.Series:
    # Weight remains zero for low-belief and reversed concepts.
    base = (
        (belief_score - 0.35)
        / 0.53
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    status_multiplier = (
        validation_status.astype(str)
        .map({
            "validated_strong": 1.00,
            "validated": 0.90,
            "direction_retained_weak": 0.62,
            "not_confirmed": 0.25,
            "insufficient_2025_sample": 0.20,
            "reversed": 0.00,
            "reversed_strong": 0.00,
        })
        .fillna(0.15)
    )

    return (
        base
        * status_multiplier
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def build_concept_beliefs(
    concepts: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    required_concept_columns = {
        "concept_id",
        "team",
        "target",
        "concept_domain",
        "concept_scope",
        "concept_name",
        "effect_direction",
        "concept_lifecycle_status",
        "concept_confidence_score",
        "weighted_lift",
        "median_absolute_lift",
        "chronological_consistency_rate",
        "member_count",
        "strong_members",
    }

    missing = (
        required_concept_columns
        - set(concepts.columns)
    )

    if missing:
        raise KeyError(
            f"Concept registry missing columns: "
            f"{sorted(missing)}"
        )

    required_validation_columns = {
        "concept_id",
        "validation_status",
        "validation_lift",
        "validation_absolute_lift",
        "direction_retained",
        "validation_p_value",
        "validation_q_value",
        "active_2025_sample",
        "inactive_2025_sample",
        "validation_games",
    }

    missing = (
        required_validation_columns
        - set(validation.columns)
    )

    if missing:
        raise KeyError(
            f"Validation registry missing columns: "
            f"{sorted(missing)}"
        )

    validation_columns = [
        "concept_id",
        "validation_status",
        "validation_lift",
        "validation_absolute_lift",
        "direction_retained",
        "validation_p_value",
        "validation_q_value",
        "active_2025_sample",
        "inactive_2025_sample",
        "active_2025_rate",
        "inactive_2025_rate",
        "validation_games",
        "available_2025_members",
        "required_active_members",
    ]

    belief = concepts.merge(
        validation[
            [
                column
                for column in validation_columns
                if column in validation.columns
            ]
        ],
        on="concept_id",
        how="left",
        validate="one_to_one",
    )

    belief["validation_status"] = (
        belief["validation_status"]
        .fillna("not_tested")
    )

    belief["discovery_confidence_component"] = (
        _clip01(
            belief[
                "concept_confidence_score"
            ]
        )
    )

    belief["discovery_status_component"] = (
        _discovery_status_component(
            belief[
                "concept_lifecycle_status"
            ]
        )
    )

    belief["discovery_effect_component"] = (
        pd.to_numeric(
            belief["median_absolute_lift"],
            errors="coerce",
        )
        .fillna(0.0)
        .div(0.20)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    belief["discovery_stability_component"] = (
        _clip01(
            belief[
                "chronological_consistency_rate"
            ]
        )
    )

    belief["member_strength_component"] = (
        (
            pd.to_numeric(
                belief["strong_members"],
                errors="coerce",
            )
            .fillna(0.0)
            + 0.35
            * (
                pd.to_numeric(
                    belief["member_count"],
                    errors="coerce",
                )
                .fillna(0.0)
                - pd.to_numeric(
                    belief["strong_members"],
                    errors="coerce",
                )
                .fillna(0.0)
            ).clip(lower=0.0)
        )
        .div(6.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    belief["validation_status_component"] = (
        _validation_status_component(
            belief["validation_status"]
        )
    )

    validation_abs_lift = (
        pd.to_numeric(
            belief[
                "validation_absolute_lift"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    belief["validation_effect_component"] = (
        validation_abs_lift
        .div(0.15)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    active_sample = pd.to_numeric(
        belief["active_2025_sample"],
        errors="coerce",
    ).fillna(0.0)

    inactive_sample = pd.to_numeric(
        belief["inactive_2025_sample"],
        errors="coerce",
    ).fillna(0.0)

    effective_sample = (
        np.minimum(
            active_sample,
            inactive_sample,
        )
    )

    belief["validation_sample_component"] = (
        effective_sample
        .div(60.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    q_value = pd.to_numeric(
        belief["validation_q_value"],
        errors="coerce",
    )

    belief["validation_significance_component"] = (
        1.0
        - q_value.fillna(1.0)
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    belief["direction_component"] = np.where(
        belief["direction_retained"]
        .fillna(False)
        .astype(bool),
        1.0,
        np.where(
            belief["validation_status"].isin(
                [
                    "reversed",
                    "reversed_strong",
                ]
            ),
            0.0,
            0.40,
        ),
    )

    # Discovery score: what ATLAS learned in 2024.
    belief["discovery_belief_score"] = (
        0.28
        * belief[
            "discovery_confidence_component"
        ]
        + 0.20
        * belief[
            "discovery_status_component"
        ]
        + 0.20
        * belief[
            "discovery_effect_component"
        ]
        + 0.20
        * belief[
            "discovery_stability_component"
        ]
        + 0.12
        * belief[
            "member_strength_component"
        ]
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # Validation score: what survived unseen 2025 games.
    belief["validation_belief_score"] = (
        0.32
        * belief[
            "validation_status_component"
        ]
        + 0.20
        * belief[
            "validation_effect_component"
        ]
        + 0.16
        * belief[
            "validation_sample_component"
        ]
        + 0.14
        * belief[
            "validation_significance_component"
        ]
        + 0.18
        * belief[
            "direction_component"
        ]
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # Blind validation carries more weight than discovery.
    belief["belief_score"] = (
        0.40
        * belief[
            "discovery_belief_score"
        ]
        + 0.60
        * belief[
            "validation_belief_score"
        ]
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # Reverse signals are retained for research but cannot
    # receive a positive 2026 prediction weight.
    reversed_mask = (
        belief["validation_status"]
        .isin(
            [
                "reversed",
                "reversed_strong",
            ]
        )
    )

    belief.loc[
        reversed_mask,
        "belief_score",
    ] = (
        belief.loc[
            reversed_mask,
            "belief_score",
        ]
        * 0.35
    )

    belief["belief_grade"] = (
        belief["belief_score"]
        .apply(_belief_grade)
    )

    belief["belief_lifecycle_stage"] = [
        _lifecycle_stage(
            validation_status=
                str(row.validation_status),
            belief_score=
                float(row.belief_score),
        )
        for row in belief.itertuples(
            index=False
        )
    ]

    belief["provisional_prediction_weight"] = (
        _prediction_weight(
            belief_score=
                belief["belief_score"],
            validation_status=
                belief["validation_status"],
        )
    )

    belief["signed_prediction_weight"] = (
        belief[
            "provisional_prediction_weight"
        ]
        * np.where(
            belief["effect_direction"].eq(
                "supports_target"
            ),
            1.0,
            -1.0,
        )
    )

    belief["prediction_weight_ready"] = (
        belief[
            "belief_lifecycle_stage"
        ].eq(
            "trusted_2026_candidate"
        )
        & belief[
            "provisional_prediction_weight"
        ].gt(0.0)
    )

    belief["league_prior_integrated"] = False
    belief["2026_outcomes_used"] = False
    belief["walk_forward_update_required"] = True
    belief["belief_engine_version"] = ENGINE_VERSION
    belief["built_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    return belief.sort_values(
        [
            "team",
            "target",
            "belief_score",
            "provisional_prediction_weight",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_belief_summary(
    belief: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        belief.groupby(
            [
                "team",
                "target",
            ],
            sort=True,
        )
        .agg(
            concepts=(
                "concept_id",
                "nunique",
            ),
            elite_beliefs=(
                "belief_grade",
                lambda values: int(
                    values.eq("elite").sum()
                ),
            ),
            strong_beliefs=(
                "belief_grade",
                lambda values: int(
                    values.eq("strong").sum()
                ),
            ),
            reliable_beliefs=(
                "belief_grade",
                lambda values: int(
                    values.eq("reliable").sum()
                ),
            ),
            possible_beliefs=(
                "belief_grade",
                lambda values: int(
                    values.eq("possible").sum()
                ),
            ),
            weak_beliefs=(
                "belief_grade",
                lambda values: int(
                    values.eq("weak").sum()
                ),
            ),
            ignored_beliefs=(
                "belief_grade",
                lambda values: int(
                    values.eq("ignore").sum()
                ),
            ),
            trusted_2026_candidates=(
                "prediction_weight_ready",
                "sum",
            ),
            mean_belief_score=(
                "belief_score",
                "mean",
            ),
            maximum_belief_score=(
                "belief_score",
                "max",
            ),
            mean_prediction_weight=(
                "provisional_prediction_weight",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["trusted_2026_candidates"] = (
        summary[
            "trusted_2026_candidates"
        ].astype("int64")
    )

    summary["discovery_season"] = (
        DISCOVERY_SEASON
    )

    summary["validation_season"] = (
        VALIDATION_SEASON
    )

    summary["2026_outcomes_used"] = False
    summary["engine_version"] = ENGINE_VERSION

    return summary


def run_concept_belief_engine() -> dict[str, Any]:
    concepts = _load_parquet(
        CONCEPT_REGISTRY_PATH,
        "2024 team concept registry",
    )

    validation = _load_parquet(
        VALIDATION_REGISTRY_PATH,
        "2025 concept validation registry",
    )

    belief = build_concept_beliefs(
        concepts=concepts,
        validation=validation,
    )

    summary = build_belief_summary(
        belief
    )

    duplicate_ids = int(
        belief[
            "concept_id"
        ].duplicated().sum()
    )

    missing_validation = int(
        belief[
            "validation_status"
        ].eq(
            "not_tested"
        ).sum()
    )

    if duplicate_ids:
        raise AssertionError(
            f"Duplicate concept IDs: {duplicate_ids}"
        )

    if belief["2026_outcomes_used"].any():
        raise AssertionError(
            "2026 outcomes were incorrectly used."
        )

    _atomic_parquet_write(
        belief,
        BELIEF_REGISTRY_PATH,
    )

    _atomic_parquet_write(
        summary,
        BELIEF_SUMMARY_PATH,
    )

    grade_counts = (
        belief["belief_grade"]
        .value_counts()
    )

    result = {
        "engine":
            "ATLAS Concept Belief Engine",
        "engine_version":
            ENGINE_VERSION,
        "discovery_season":
            DISCOVERY_SEASON,
        "validation_season":
            VALIDATION_SEASON,
        "concepts_scored":
            int(len(belief)),
        "teams":
            int(
                belief["team"].nunique()
            ),
        "targets":
            int(
                belief["target"].nunique()
            ),
        "elite":
            int(
                grade_counts.get(
                    "elite",
                    0,
                )
            ),
        "strong":
            int(
                grade_counts.get(
                    "strong",
                    0,
                )
            ),
        "reliable":
            int(
                grade_counts.get(
                    "reliable",
                    0,
                )
            ),
        "possible":
            int(
                grade_counts.get(
                    "possible",
                    0,
                )
            ),
        "weak":
            int(
                grade_counts.get(
                    "weak",
                    0,
                )
            ),
        "ignore":
            int(
                grade_counts.get(
                    "ignore",
                    0,
                )
            ),
        "trusted_2026_candidates":
            int(
                belief[
                    "prediction_weight_ready"
                ].sum()
            ),
        "missing_validation":
            missing_validation,
        "duplicate_concept_ids":
            duplicate_ids,
        "league_prior_integrated":
            False,
        "2026_outcomes_used":
            False,
        "outputs": {
            "belief_registry": str(
                BELIEF_REGISTRY_PATH
            ),
            "belief_summary": str(
                BELIEF_SUMMARY_PATH
            ),
        },
        "policy": {
            "belief_is_continuous":
                True,
            "binary_validation_gate":
                False,
            "reversed_concepts_receive_weight":
                False,
            "weights_are_provisional":
                True,
            "league_prior_integration_pending":
                True,
            "2026_walk_forward_updates_required":
                True,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS CONCEPT BELIEF ENGINE")
    print("=" * 78)
    print(
        f"Concepts Scored............ "
        f"{len(belief):,}"
    )
    print(
        f"Teams...................... "
        f"{belief['team'].nunique():,}"
    )
    print(
        f"Targets.................... "
        f"{belief['target'].nunique():,}"
    )
    print(
        f"Elite Beliefs.............. "
        f"{result['elite']:,}"
    )
    print(
        f"Strong Beliefs............. "
        f"{result['strong']:,}"
    )
    print(
        f"Reliable Beliefs........... "
        f"{result['reliable']:,}"
    )
    print(
        f"Possible Beliefs........... "
        f"{result['possible']:,}"
    )
    print(
        f"Weak Beliefs............... "
        f"{result['weak']:,}"
    )
    print(
        f"Ignored Beliefs............ "
        f"{result['ignore']:,}"
    )
    print(
        f"Trusted 2026 Candidates.... "
        f"{result['trusted_2026_candidates']:,}"
    )
    print(
        f"Missing Validation......... "
        f"{missing_validation:,}"
    )
    print(
        f"2026 Outcomes Used......... "
        f"{result['2026_outcomes_used']}"
    )
    print(
        f"Saved To................... "
        f"{BELIEF_REGISTRY_PATH}"
    )
    print("=" * 78)

    return result
