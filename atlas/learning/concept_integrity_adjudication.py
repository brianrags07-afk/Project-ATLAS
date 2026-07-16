"""
Concept-integrity adjudication for Project ATLAS.

This module preserves raw two-member discovery concepts while identifying:

- exact joint activation-mask duplicates,
- inverse or complementary joint masks,
- concepts formed from related transformations of the same underlying fact,
- concepts formed entirely within one broad evidence domain,
- mirrored team/opponent identity constructions,
- concepts with insufficient structural independence.

No prediction weight, probability, confidence value, or prediction is created.
"""

from __future__ import annotations

from typing import Final
import hashlib
import re

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.1"


def broad_evidence_domain(
    source_classification: str,
) -> str:
    value = str(
        source_classification
    ).upper()

    if "BULLPEN" in value:
        return "BULLPEN"

    if (
        "LINEUP" in value
        or "STARTER" in value
    ):
        return "LINEUP_STARTER"

    if "IDENTITY" in value:
        return "IDENTITY"

    if "AVAILABILITY" in value:
        return "AVAILABILITY"

    return "OTHER"


def remove_game_side_prefix(
    feature_name: str,
) -> str:
    value = str(
        feature_name
    )

    for prefix in [
        "home__",
        "away__",
    ]:
        if value.startswith(
            prefix
        ):
            return value[
                len(prefix):
            ]

    return value


def feature_lineage_root(
    feature_name: str,
) -> str:
    """
    Create a conservative root describing the underlying factual measurement.

    This intentionally collapses:

    - team/opponent identity perspectives,
    - identity edge wrappers,
    - home/away wrappers,
    - common mirror words such as team/opponent and won/lost.

    It does not remove the actual metric family such as run differential,
    maximum lead, bullpen walks, or lineup home-run rate.
    """

    value = remove_game_side_prefix(
        feature_name
    ).lower()

    prefixes = [
        "identity__identity_edge__",
        "identity__team_identity__",
        "identity__opponent_identity__",
        "identity__",
        "bullpen__",
        "lineup_starter__",
    ]

    source_prefix = ""

    for prefix in prefixes:
        if value.startswith(
            prefix
        ):
            source_prefix = prefix.rstrip(
                "_"
            ).split(
                "__"
            )[0]

            value = value[
                len(prefix):
            ]

            break

    replacements = {
        "opponent_scoring_run": "scoring_run",
        "team_scoring_run": "scoring_run",
        "opponent_score": "score",
        "team_score": "score",
        "opponent_runs": "runs",
        "team_runs": "runs",
        "opponent_": "",
        "team_": "",
        "winner_": "",
        "loser_": "",
        "won_by_": "margin_",
        "lost_by_": "margin_",
        "win_by_": "margin_",
        "loss_by_": "margin_",
        "covered_minus_1_5": "margin_2_plus",
        "covered_plus_1_5": "margin_2_plus",
        "failed_minus_1_5": "margin_2_plus",
        "held_minus_1_5": "margin_2_plus",
        "prior_mean": "historical_mean",
        "career_prior": "historical",
        "season_prior": "historical",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    value = re.sub(
        r"(^|_)won($|_)",
        r"\1result\2",
        value,
    )

    value = re.sub(
        r"(^|_)lost($|_)",
        r"\1result\2",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip(
        "_"
    )

    return (
        source_prefix
        + "::"
        + value
    )


def underlying_metric_root(
    feature_name: str,
) -> str:
    """
    Return the lineage root without its source wrapper.

    This recognizes mirrored team, opponent, and identity-edge versions of
    the same underlying measurement.
    """

    lineage = feature_lineage_root(
        feature_name
    )

    if "::" in lineage:
        return lineage.split(
            "::",
            1,
        )[1]

    return lineage


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
        f"Unsupported threshold operator: {operator}"
    )


def trinary_joint_state(
    dataframe: pd.DataFrame,
    member_1_feature: str,
    member_1_operator: str,
    member_1_threshold: float,
    member_2_feature: str,
    member_2_operator: str,
    member_2_threshold: float,
) -> np.ndarray:
    """
    Encode the joint concept state.

    0 = at least one member unavailable.
    1 = both members available but concept inactive.
    2 = both members available and both conditions active.
    """

    for feature_name in [
        member_1_feature,
        member_2_feature,
    ]:
        if feature_name not in dataframe.columns:
            raise KeyError(
                f"Concept member feature missing from view: {feature_name}"
            )

    first_numeric = pd.to_numeric(
        dataframe[
            member_1_feature
        ],
        errors="coerce",
    )

    second_numeric = pd.to_numeric(
        dataframe[
            member_2_feature
        ],
        errors="coerce",
    )

    available = (
        first_numeric.notna()
        & second_numeric.notna()
    )

    first_active = condition_mask(
        series=first_numeric,
        operator=member_1_operator,
        threshold=float(
            member_1_threshold
        ),
    )

    second_active = condition_mask(
        series=second_numeric,
        operator=member_2_operator,
        threshold=float(
            member_2_threshold
        ),
    )

    joint_active = (
        available
        & first_active
        & second_active
    )

    state = np.zeros(
        len(dataframe),
        dtype=np.uint8,
    )

    state[
        available.to_numpy()
    ] = 1

    state[
        joint_active.to_numpy()
    ] = 2

    return state


def complement_trinary_state(
    state: np.ndarray,
) -> np.ndarray:
    output = np.asarray(
        state,
        dtype=np.uint8,
    ).copy()

    available = output > 0

    output[
        available
    ] = (
        3
        - output[
            available
        ]
    )

    return output


def hash_state(
    state: np.ndarray,
) -> str:
    return hashlib.sha256(
        np.asarray(
            state,
            dtype=np.uint8,
        ).tobytes()
    ).hexdigest()


def undirected_joint_mask_key(
    mask_hash: str,
    complement_hash: str,
) -> str:
    return min(
        str(mask_hash),
        str(complement_hash),
    )


def lineage_pair_key(
    first_root: str,
    second_root: str,
) -> str:
    return " + ".join(
        sorted(
            [
                str(first_root),
                str(second_root),
            ]
        )
    )


def broad_domain_pair_key(
    first_domain: str,
    second_domain: str,
) -> str:
    return " + ".join(
        sorted(
            [
                str(first_domain),
                str(second_domain),
            ]
        )
    )


def concept_structural_status(
    same_underlying_metric: bool,
    same_broad_domain: bool,
    mirrored_identity_pair: bool,
    exact_joint_duplicate: bool,
    inverse_joint_duplicate: bool,
    nominated_representative: bool,
) -> str:
    if same_underlying_metric:
        return "BLOCK_SHARED_UNDERLYING_METRIC"

    if mirrored_identity_pair:
        return "BLOCK_MIRRORED_IDENTITY_CONSTRUCTION"

    if same_broad_domain:
        return "BLOCK_SINGLE_BROAD_DOMAIN"

    if not nominated_representative:
        if exact_joint_duplicate:
            return "REDUNDANT_EXACT_JOINT_MASK"

        if inverse_joint_duplicate:
            return "REDUNDANT_INVERSE_JOINT_MASK"

        return "REDUNDANT_JOINT_MASK_MEMBER"

    return "NOMINATED_INDEPENDENT_CONCEPT"


def add_representative_ranking(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output[
        "_priority_structural_block"
    ] = (
        output[
            "same_underlying_metric"
        ]
        | output[
            "same_broad_domain"
        ]
        | output[
            "mirrored_identity_pair"
        ]
    ).astype(
        int
    )

    status_rank = {
        "STRONG_CONCEPT_CANDIDATE": 0,
        "CONCEPT_CANDIDATE": 1,
        "WEAK_CONCEPT_CANDIDATE": 2,
    }

    output[
        "_priority_status"
    ] = output[
        "concept_status"
    ].map(
        status_rank
    ).fillna(
        9
    )

    output[
        "_priority_q"
    ] = pd.to_numeric(
        output[
            "q_value"
        ],
        errors="coerce",
    ).fillna(
        np.inf
    )

    output[
        "_priority_incremental"
    ] = -pd.to_numeric(
        output[
            "incremental_lift_over_strongest_member"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_priority_joint_lift"
    ] = -pd.to_numeric(
        output[
            "absolute_joint_lift"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_priority_sample"
    ] = -pd.to_numeric(
        output[
            "joint_active_sample"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    return output


def nominate_joint_mask_representatives(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    ranked = add_representative_ranking(
        dataframe
    )

    ranked = ranked.sort_values(
        [
            "target_name",
            "undirected_joint_mask_key",
            "_priority_structural_block",
            "_priority_status",
            "_priority_q",
            "_priority_incremental",
            "_priority_joint_lift",
            "_priority_sample",
            "concept_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
    )

    ranked[
        "joint_mask_group_rank"
    ] = (
        ranked.groupby(
            [
                "target_name",
                "undirected_joint_mask_key",
            ],
            sort=False,
        )
        .cumcount()
        + 1
    )

    ranked[
        "nominated_joint_mask_representative"
    ] = ranked[
        "joint_mask_group_rank"
    ].eq(
        1
    )

    return ranked.drop(
        columns=[
            "_priority_structural_block",
            "_priority_status",
            "_priority_q",
            "_priority_incremental",
            "_priority_joint_lift",
            "_priority_sample",
        ],
        errors="ignore",
    )
