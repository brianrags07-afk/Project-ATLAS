"""
Controlled cross-feature concept formation for Project ATLAS.

This module forms factual two-member research concepts from nominated,
non-analogue univariate evidence representatives.

Concepts are allowed only when:

- both members belong to the same target,
- members come from different governed source domains,
- neither member is a direct historical target analogue,
- neither member is an availability artifact,
- activation masks are not near-duplicates,
- the joint condition has sufficient active and inactive samples,
- the joint condition improves on the stronger individual member.

The module creates research concepts only.

It does not assign prediction weights, produce probabilities, create
predictions, or modify canonical evidence.
"""

from __future__ import annotations

from typing import Final
import hashlib
import math

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

MIN_MEMBER_ACTIVE_SAMPLE: Final[int] = 20
MIN_JOINT_ACTIVE_SAMPLE: Final[int] = 25
MIN_JOINT_INACTIVE_SAMPLE: Final[int] = 25
MAX_JACCARD_OVERLAP: Final[float] = 0.85
MAX_ABSOLUTE_PHI: Final[float] = 0.85
MIN_INCREMENTAL_LIFT: Final[float] = 0.015


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

    output = pd.Series(
        np.nan,
        index=p_values.index,
        dtype="float64",
    )

    if not valid.any():
        return output

    ordered_indices = numeric[
        valid
    ].sort_values(
        kind="stable"
    ).index

    ordered_p = numeric.loc[
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

    adjusted = np.minimum.accumulate(
        raw_q[
            ::-1
        ]
    )[
        ::-1
    ]

    output.loc[
        ordered_indices
    ] = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    return output


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


def reconstruct_member_masks(
    dataframe: pd.DataFrame,
    member: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    feature_name = str(
        member[
            "feature_name"
        ]
    )

    if feature_name not in dataframe.columns:
        raise KeyError(
            f"Feature missing from controlled view: {feature_name}"
        )

    numeric = pd.to_numeric(
        dataframe[
            feature_name
        ],
        errors="coerce",
    )

    available = numeric.notna()

    active = (
        condition_mask(
            series=numeric,
            operator=str(
                member[
                    "threshold_operator"
                ]
            ),
            threshold=float(
                member[
                    "threshold_value"
                ]
            ),
        )
        & available
    )

    return (
        available.astype(
            bool
        ),
        active.astype(
            bool
        ),
    )


def jaccard_similarity(
    first: pd.Series,
    second: pd.Series,
) -> float:
    first_bool = first.astype(
        bool
    ).to_numpy()

    second_bool = second.astype(
        bool
    ).to_numpy()

    intersection = int(
        np.logical_and(
            first_bool,
            second_bool,
        ).sum()
    )

    union = int(
        np.logical_or(
            first_bool,
            second_bool,
        ).sum()
    )

    if union == 0:
        return 0.0

    return float(
        intersection
        / union
    )


def phi_coefficient(
    first: pd.Series,
    second: pd.Series,
    valid_mask: pd.Series | None = None,
) -> float:
    first_bool = first.astype(
        bool
    )

    second_bool = second.astype(
        bool
    )

    if valid_mask is not None:
        valid = valid_mask.astype(
            bool
        )

        first_bool = first_bool[
            valid
        ]

        second_bool = second_bool[
            valid
        ]

    a = int(
        (
            first_bool
            & second_bool
        ).sum()
    )

    b = int(
        (
            first_bool
            & ~second_bool
        ).sum()
    )

    c = int(
        (
            ~first_bool
            & second_bool
        ).sum()
    )

    d = int(
        (
            ~first_bool
            & ~second_bool
        ).sum()
    )

    denominator = math.sqrt(
        float(
            (a + b)
            * (c + d)
            * (a + c)
            * (b + d)
        )
    )

    if denominator == 0:
        return 0.0

    return float(
        (
            (a * d)
            - (b * c)
        )
        / denominator
    )


def stable_concept_id(
    target_name: str,
    first_member_id: str,
    second_member_id: str,
) -> str:
    ordered_members = sorted(
        [
            str(
                first_member_id
            ),
            str(
                second_member_id
            ),
        ]
    )

    payload = (
        str(
            target_name
        )
        + "||"
        + "||".join(
            ordered_members
        )
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[
        :20
    ]

    return (
        f"{target_name}__concept__{digest}"
    )


def member_identifier(
    row: pd.Series,
) -> str:
    return (
        str(
            row[
                "feature_name"
            ]
        )
        + "::"
        + str(
            row[
                "condition_name"
            ]
        )
        + "::"
        + str(
            row[
                "threshold_operator"
            ]
        )
        + "::"
        + format(
            float(
                row[
                    "threshold_value"
                ]
            ),
            ".12g",
        )
    )


def pair_domain_key(
    first_domain: str,
    second_domain: str,
) -> str:
    return " + ".join(
        sorted(
            [
                str(
                    first_domain
                ),
                str(
                    second_domain
                ),
            ]
        )
    )


def concept_status(
    joint_active_sample: int,
    joint_inactive_sample: int,
    absolute_joint_lift: float,
    incremental_lift: float,
    q_value: float | None,
) -> str:
    if (
        joint_active_sample < MIN_JOINT_ACTIVE_SAMPLE
        or joint_inactive_sample < MIN_JOINT_INACTIVE_SAMPLE
    ):
        return "INSUFFICIENT_JOINT_SAMPLE"

    if (
        q_value is None
        or pd.isna(
            q_value
        )
    ):
        return "DESCRIPTIVE_ONLY"

    if incremental_lift < MIN_INCREMENTAL_LIFT:
        return "NO_INCREMENTAL_VALUE"

    if (
        q_value <= 0.01
        and absolute_joint_lift >= 0.08
        and incremental_lift >= 0.03
    ):
        return "STRONG_CONCEPT_CANDIDATE"

    if (
        q_value <= 0.05
        and absolute_joint_lift >= 0.05
        and incremental_lift >= 0.02
    ):
        return "CONCEPT_CANDIDATE"

    if (
        q_value <= 0.10
        and absolute_joint_lift >= 0.025
        and incremental_lift >= MIN_INCREMENTAL_LIFT
    ):
        return "WEAK_CONCEPT_CANDIDATE"

    return "NOT_CONFIRMED"


def evaluate_pair(
    dataframe: pd.DataFrame,
    first_member: pd.Series,
    second_member: pd.Series,
    target_name: str,
) -> dict[str, object]:
    first_available, first_active = reconstruct_member_masks(
        dataframe=dataframe,
        member=first_member,
    )

    second_available, second_active = reconstruct_member_masks(
        dataframe=dataframe,
        member=second_member,
    )

    pair_available = (
        first_available
        & second_available
    )

    first_active_available = (
        first_active
        & pair_available
    )

    second_active_available = (
        second_active
        & pair_available
    )

    joint_active = (
        first_active_available
        & second_active_available
    )

    joint_inactive = (
        pair_available
        & ~joint_active
    )

    target = dataframe[
        "target_label"
    ].astype(
        bool
    )

    first_active_sample = int(
        first_active_available.sum()
    )

    second_active_sample = int(
        second_active_available.sum()
    )

    joint_active_sample = int(
        joint_active.sum()
    )

    joint_inactive_sample = int(
        joint_inactive.sum()
    )

    first_successes = int(
        target[
            first_active_available
        ].sum()
    )

    second_successes = int(
        target[
            second_active_available
        ].sum()
    )

    joint_successes = int(
        target[
            joint_active
        ].sum()
    )

    joint_inactive_successes = int(
        target[
            joint_inactive
        ].sum()
    )

    first_rate = (
        first_successes
        / first_active_sample
        if first_active_sample
        else np.nan
    )

    second_rate = (
        second_successes
        / second_active_sample
        if second_active_sample
        else np.nan
    )

    joint_rate = (
        joint_successes
        / joint_active_sample
        if joint_active_sample
        else np.nan
    )

    joint_inactive_rate = (
        joint_inactive_successes
        / joint_inactive_sample
        if joint_inactive_sample
        else np.nan
    )

    joint_lift = (
        joint_rate
        - joint_inactive_rate
        if (
            joint_active_sample
            and joint_inactive_sample
        )
        else np.nan
    )

    first_direction = str(
        first_member[
            "effect_direction"
        ]
    )

    second_direction = str(
        second_member[
            "effect_direction"
        ]
    )

    same_effect_direction = (
        first_direction
        == second_direction
    )

    if first_direction == "SUPPORTS_TARGET":
        strongest_member_rate = max(
            first_rate,
            second_rate,
        )

        incremental_lift = (
            joint_rate
            - strongest_member_rate
        )

    elif first_direction == "OPPOSES_TARGET":
        strongest_member_rate = min(
            first_rate,
            second_rate,
        )

        incremental_lift = (
            strongest_member_rate
            - joint_rate
        )

    else:
        strongest_member_rate = np.nan
        incremental_lift = np.nan

    p_value = two_proportion_p_value(
        active_successes=joint_successes,
        active_sample=joint_active_sample,
        inactive_successes=joint_inactive_successes,
        inactive_sample=joint_inactive_sample,
    )

    jaccard = jaccard_similarity(
        first_active_available,
        second_active_available,
    )

    phi = phi_coefficient(
        first_active_available,
        second_active_available,
        valid_mask=pair_available,
    )

    first_member_id = member_identifier(
        first_member
    )

    second_member_id = member_identifier(
        second_member
    )

    return {
        "concept_id":
            stable_concept_id(
                target_name=target_name,
                first_member_id=first_member_id,
                second_member_id=second_member_id,
            ),

        "target_name":
            target_name,

        "discovery_grain":
            str(
                first_member[
                    "discovery_grain"
                ]
            ),

        "member_1_id":
            first_member_id,

        "member_2_id":
            second_member_id,

        "member_1_feature":
            str(
                first_member[
                    "feature_name"
                ]
            ),

        "member_2_feature":
            str(
                second_member[
                    "feature_name"
                ]
            ),

        "member_1_domain":
            str(
                first_member[
                    "source_classification"
                ]
            ),

        "member_2_domain":
            str(
                second_member[
                    "source_classification"
                ]
            ),

        "domain_pair":
            pair_domain_key(
                str(
                    first_member[
                        "source_classification"
                    ]
                ),
                str(
                    second_member[
                        "source_classification"
                    ]
                ),
            ),

        "member_1_effect_direction":
            first_direction,

        "member_2_effect_direction":
            second_direction,

        "same_effect_direction":
            same_effect_direction,

        "member_1_active_sample":
            first_active_sample,

        "member_2_active_sample":
            second_active_sample,

        "member_1_active_rate":
            first_rate,

        "member_2_active_rate":
            second_rate,

        "pair_available_sample":
            int(
                pair_available.sum()
            ),

        "joint_active_sample":
            joint_active_sample,

        "joint_active_successes":
            joint_successes,

        "joint_active_rate":
            joint_rate,

        "joint_inactive_sample":
            joint_inactive_sample,

        "joint_inactive_successes":
            joint_inactive_successes,

        "joint_inactive_rate":
            joint_inactive_rate,

        "joint_lift":
            joint_lift,

        "absolute_joint_lift":
            (
                abs(
                    joint_lift
                )
                if not pd.isna(
                    joint_lift
                )
                else np.nan
            ),

        "strongest_member_active_rate":
            strongest_member_rate,

        "incremental_lift_over_strongest_member":
            incremental_lift,

        "jaccard_overlap":
            jaccard,

        "phi_coefficient":
            phi,

        "absolute_phi":
            abs(
                phi
            ),

        "p_value":
            p_value,

        "prediction_weight_assigned":
            False,

        "prediction_created":
            False,

        "validation_season_used":
            False,

        "engine_version":
            ENGINE_VERSION,
    }
