"""
Univariate factual evidence discovery for Project ATLAS.

Each governed pregame feature is evaluated independently against one factual
target at a time.

This module reports:

- available sample,
- missing sample,
- distribution-derived activation conditions,
- active and inactive samples,
- active and inactive outcome rates,
- lift,
- relative risk,
- odds ratio,
- two-proportion p-value,
- Benjamini-Hochberg q-value,
- factual research status.

It does not assign prediction weights, create predictions, combine evidence,
or alter canonical pregame evidence.
"""

from __future__ import annotations

from typing import Final
import math

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

NON_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
    "home_team",
    "away_team",
    "target_label",
    "target_name",
    "discovery_grain",
    "discovery_feature_count",
    "discovery_available_feature_count",
    "discovery_missing_feature_count",
    "discovery_feature_coverage_rate",
    "strict_backtest_safe",
    "shared_game_target_counted_once",
    "canonical_evidence_modified",
    "prediction_created",
    "weight_assigned",
    "discovery_view_version",
)

FORBIDDEN_FEATURE_TOKENS: Final[tuple[str, ...]] = (
    "target_",
    "actual_",
    "final_score",
    "prediction",
    "weight_assigned",
    "game_total_runs",
    "team_runs",
    "opponent_runs",
    "run_margin",
)

MIN_AVAILABLE_SAMPLE: Final[int] = 40
MIN_ACTIVE_SAMPLE: Final[int] = 20
MIN_INACTIVE_SAMPLE: Final[int] = 20


def two_proportion_p_value(
    active_successes: int,
    active_sample: int,
    inactive_successes: int,
    inactive_sample: int,
) -> float | None:
    if (
        active_sample <= 0
        or inactive_sample <= 0
    ):
        return None

    active_rate = (
        active_successes
        / active_sample
    )

    inactive_rate = (
        inactive_successes
        / inactive_sample
    )

    pooled_rate = (
        (
            active_successes
            + inactive_successes
        )
        / (
            active_sample
            + inactive_sample
        )
    )

    variance = (
        pooled_rate
        * (
            1.0
            - pooled_rate
        )
        * (
            (1.0 / active_sample)
            + (1.0 / inactive_sample)
        )
    )

    if variance <= 0:
        return 1.0

    z_score = (
        active_rate
        - inactive_rate
    ) / math.sqrt(
        variance
    )

    return float(
        math.erfc(
            abs(
                z_score
            )
            / math.sqrt(
                2.0
            )
        )
    )


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        p_values,
        errors="coerce",
    )

    valid = numeric.notna()

    q_values = pd.Series(
        np.nan,
        index=p_values.index,
        dtype="float64",
    )

    if not valid.any():
        return q_values

    valid_values = numeric[
        valid
    ]

    ordered_indices = valid_values.sort_values(
        kind="stable"
    ).index

    ordered_p = valid_values.loc[
        ordered_indices
    ].to_numpy(
        dtype=float
    )

    count = len(
        ordered_p
    )

    raw_q = (
        ordered_p
        * count
        / np.arange(
            1,
            count + 1,
            dtype=float,
        )
    )

    adjusted_q = np.minimum.accumulate(
        raw_q[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted_q = np.clip(
        adjusted_q,
        0.0,
        1.0,
    )

    q_values.loc[
        ordered_indices
    ] = adjusted_q

    return q_values


def feature_family(
    feature_name: str,
) -> str:
    name = str(
        feature_name
    )

    if name.startswith(
        "home__"
    ):
        name = name[
            len(
                "home__"
            ):
        ]

    elif name.startswith(
        "away__"
    ):
        name = name[
            len(
                "away__"
            ):
        ]

    return (
        name.split(
            "__",
            1,
        )[0]
        if "__" in name
        else "other"
    )


def feature_side(
    feature_name: str,
) -> str:
    if feature_name.startswith(
        "home__"
    ):
        return "HOME"

    if feature_name.startswith(
        "away__"
    ):
        return "AWAY"

    return "TEAM"


def is_eligible_feature(
    dataframe: pd.DataFrame,
    column: str,
) -> tuple[bool, str]:
    if column in NON_FEATURE_COLUMNS:
        return (
            False,
            "NON_FEATURE_CONTEXT_OR_GOVERNANCE",
        )

    lower = column.lower()

    if any(
        token in lower
        for token in FORBIDDEN_FEATURE_TOKENS
    ):
        return (
            False,
            "FORBIDDEN_OUTCOME_OR_DECISION_TOKEN",
        )

    series = dataframe[
        column
    ]

    if not (
        pd.api.types.is_numeric_dtype(
            series
        )
        or pd.api.types.is_bool_dtype(
            series
        )
    ):
        return (
            False,
            "NON_NUMERIC_FEATURE",
        )

    available = int(
        series.notna().sum()
    )

    if available < MIN_AVAILABLE_SAMPLE:
        return (
            False,
            "INSUFFICIENT_AVAILABLE_SAMPLE",
        )

    unique_values = int(
        series.dropna().nunique()
    )

    if unique_values < 2:
        return (
            False,
            "CONSTANT_FEATURE",
        )

    return (
        True,
        "ELIGIBLE",
    )


def numeric_conditions(
    series: pd.Series,
) -> list[dict[str, object]]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    available = numeric.dropna()

    if available.empty:
        return []

    unique_values = int(
        available.nunique()
    )

    if unique_values < 2:
        return []

    conditions = []

    if unique_values == 2:
        sorted_values = sorted(
            available.unique().tolist()
        )

        high_value = sorted_values[
            -1
        ]

        conditions.append({
            "condition_name":
                "equals_high_value",

            "operator":
                "==",

            "threshold":
                float(
                    high_value
                ),

            "quantile":
                None,
        })

        return conditions

    quantile_specs = [
        (
            "lower_quartile",
            "<=",
            0.25,
        ),
        (
            "upper_quartile",
            ">=",
            0.75,
        ),
    ]

    for condition_name, operator, quantile in quantile_specs:
        threshold = float(
            available.quantile(
                quantile
            )
        )

        conditions.append({
            "condition_name":
                condition_name,

            "operator":
                operator,

            "threshold":
                threshold,

            "quantile":
                quantile,
        })

    return conditions


def condition_mask(
    series: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if operator == ">=":
        return numeric.ge(
            threshold
        )

    if operator == "<=":
        return numeric.le(
            threshold
        )

    if operator == "==":
        return numeric.eq(
            threshold
        )

    raise ValueError(
        f"Unsupported operator: {operator}"
    )


def research_status(
    available_sample: int,
    active_sample: int,
    inactive_sample: int,
    absolute_lift: float | None,
    q_value: float | None,
) -> str:
    if (
        available_sample < MIN_AVAILABLE_SAMPLE
        or active_sample < MIN_ACTIVE_SAMPLE
        or inactive_sample < MIN_INACTIVE_SAMPLE
    ):
        return "INSUFFICIENT_SAMPLE"

    if (
        absolute_lift is None
        or pd.isna(
            absolute_lift
        )
        or q_value is None
        or pd.isna(
            q_value
        )
    ):
        return "DESCRIPTIVE_ONLY"

    if (
        q_value <= 0.01
        and absolute_lift >= 0.08
    ):
        return "STRONG_DISCOVERY_CANDIDATE"

    if (
        q_value <= 0.05
        and absolute_lift >= 0.05
    ):
        return "DISCOVERY_CANDIDATE"

    if (
        q_value <= 0.10
        and absolute_lift >= 0.025
    ):
        return "WEAK_DISCOVERY_CANDIDATE"

    return "NOT_CONFIRMED"


def discover_feature(
    dataframe: pd.DataFrame,
    feature_name: str,
    target_name: str,
    discovery_grain: str,
) -> list[dict[str, object]]:
    target = dataframe[
        "target_label"
    ].astype(
        bool
    )

    feature = dataframe[
        feature_name
    ]

    available_mask = feature.notna()

    available_sample = int(
        available_mask.sum()
    )

    missing_sample = int(
        (
            ~available_mask
        ).sum()
    )

    baseline_successes = int(
        target[
            available_mask
        ].sum()
    )

    baseline_rate = (
        baseline_successes
        / available_sample
        if available_sample
        else np.nan
    )

    records = []

    for condition in numeric_conditions(
        feature
    ):
        active_mask = (
            available_mask
            & condition_mask(
                feature,
                operator=str(
                    condition[
                        "operator"
                    ]
                ),
                threshold=float(
                    condition[
                        "threshold"
                    ]
                ),
            )
        )

        inactive_mask = (
            available_mask
            & ~active_mask
        )

        active_sample = int(
            active_mask.sum()
        )

        inactive_sample = int(
            inactive_mask.sum()
        )

        active_successes = int(
            target[
                active_mask
            ].sum()
        )

        inactive_successes = int(
            target[
                inactive_mask
            ].sum()
        )

        active_rate = (
            active_successes
            / active_sample
            if active_sample
            else np.nan
        )

        inactive_rate = (
            inactive_successes
            / inactive_sample
            if inactive_sample
            else np.nan
        )

        lift = (
            active_rate
            - inactive_rate
            if (
                active_sample
                and inactive_sample
            )
            else np.nan
        )

        absolute_lift = (
            abs(
                lift
            )
            if not pd.isna(
                lift
            )
            else np.nan
        )

        relative_risk = (
            active_rate
            / inactive_rate
            if (
                inactive_rate is not None
                and not pd.isna(
                    inactive_rate
                )
                and inactive_rate > 0
            )
            else np.nan
        )

        active_failures = (
            active_sample
            - active_successes
        )

        inactive_failures = (
            inactive_sample
            - inactive_successes
        )

        if (
            active_successes > 0
            and active_failures > 0
            and inactive_successes > 0
            and inactive_failures > 0
        ):
            odds_ratio = (
                (
                    active_successes
                    / active_failures
                )
                / (
                    inactive_successes
                    / inactive_failures
                )
            )

        else:
            odds_ratio = np.nan

        p_value = two_proportion_p_value(
            active_successes=active_successes,
            active_sample=active_sample,
            inactive_successes=inactive_successes,
            inactive_sample=inactive_sample,
        )

        if pd.isna(
            lift
        ):
            effect_direction = "UNAVAILABLE"

        elif lift > 0:
            effect_direction = "SUPPORTS_TARGET"

        elif lift < 0:
            effect_direction = "OPPOSES_TARGET"

        else:
            effect_direction = "NEUTRAL"

        records.append({
            "target_name":
                target_name,

            "discovery_grain":
                discovery_grain,

            "feature_name":
                feature_name,

            "feature_family":
                feature_family(
                    feature_name
                ),

            "feature_side":
                feature_side(
                    feature_name
                ),

            "feature_dtype":
                str(
                    feature.dtype
                ),

            "condition_name":
                condition[
                    "condition_name"
                ],

            "threshold_operator":
                condition[
                    "operator"
                ],

            "threshold_value":
                condition[
                    "threshold"
                ],

            "threshold_quantile":
                condition[
                    "quantile"
                ],

            "total_rows":
                int(
                    len(
                        dataframe
                    )
                ),

            "available_sample":
                available_sample,

            "missing_sample":
                missing_sample,

            "baseline_successes":
                baseline_successes,

            "baseline_rate":
                baseline_rate,

            "active_sample":
                active_sample,

            "active_successes":
                active_successes,

            "active_rate":
                active_rate,

            "inactive_sample":
                inactive_sample,

            "inactive_successes":
                inactive_successes,

            "inactive_rate":
                inactive_rate,

            "lift":
                lift,

            "absolute_lift":
                absolute_lift,

            "relative_risk":
                relative_risk,

            "odds_ratio":
                odds_ratio,

            "p_value":
                p_value,

            "effect_direction":
                effect_direction,

            "prediction_weight_assigned":
                False,

            "prediction_created":
                False,

            "engine_version":
                ENGINE_VERSION,
        })

    return records


def discover_target(
    dataframe: pd.DataFrame,
    target_name: str,
    discovery_grain: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    required = {
        "target_label",
        "target_name",
        "discovery_grain",
    }

    missing = sorted(
        required.difference(
            dataframe.columns
        )
    )

    if missing:
        raise KeyError(
            f"Discovery view lacks required columns: {missing}"
        )

    if not pd.api.types.is_bool_dtype(
        dataframe[
            "target_label"
        ]
    ):
        raise TypeError(
            "target_label must be boolean."
        )

    audit_rows = []
    records = []

    for feature_name in dataframe.columns:
        eligible, reason = is_eligible_feature(
            dataframe,
            feature_name,
        )

        audit_rows.append({
            "target_name":
                target_name,

            "feature_name":
                feature_name,

            "eligible":
                eligible,

            "reason":
                reason,

            "dtype":
                str(
                    dataframe[
                        feature_name
                    ].dtype
                ),

            "available_sample":
                int(
                    dataframe[
                        feature_name
                    ].notna().sum()
                ),
        })

        if not eligible:
            continue

        records.extend(
            discover_feature(
                dataframe=dataframe,
                feature_name=feature_name,
                target_name=target_name,
                discovery_grain=discovery_grain,
            )
        )

    registry = pd.DataFrame(
        records
    )

    audit = pd.DataFrame(
        audit_rows
    )

    if registry.empty:
        return (
            registry,
            audit,
        )

    registry[
        "q_value"
    ] = benjamini_hochberg(
        registry[
            "p_value"
        ]
    )

    registry[
        "research_status"
    ] = [
        research_status(
            available_sample=int(
                row.available_sample
            ),
            active_sample=int(
                row.active_sample
            ),
            inactive_sample=int(
                row.inactive_sample
            ),
            absolute_lift=(
                float(
                    row.absolute_lift
                )
                if not pd.isna(
                    row.absolute_lift
                )
                else None
            ),
            q_value=(
                float(
                    row.q_value
                )
                if not pd.isna(
                    row.q_value
                )
                else None
            ),
        )
        for row in registry.itertuples(
            index=False
        )
    ]

    registry = registry.sort_values(
        [
            "research_status",
            "q_value",
            "absolute_lift",
            "feature_name",
            "condition_name",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return (
        registry,
        audit,
    )
