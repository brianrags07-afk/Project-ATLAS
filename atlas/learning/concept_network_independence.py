"""
Concept-network independence governance for Project ATLAS.

The engine identifies concept candidates that are not truly independent because
they:

- activate in nearly the same historical games,
- repeatedly reuse the same condition member,
- differ only through closely related lineage members,
- form connected networks of near-equivalent explanations.

No prediction weight, probability, confidence adjustment, or prediction is
created.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


ENGINE_VERSION = "1.0.0"


class UnionFind:
    def __init__(self, values):
        self.parent = {
            value: value
            for value in values
        }

        self.rank = {
            value: 0
            for value in values
        }

    def find(self, value):
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(
                parent
            )

        return self.parent[value]

    def union(self, first, second):
        root_first = self.find(
            first
        )

        root_second = self.find(
            second
        )

        if root_first == root_second:
            return

        rank_first = self.rank[
            root_first
        ]

        rank_second = self.rank[
            root_second
        ]

        if rank_first < rank_second:
            self.parent[
                root_first
            ] = root_second

        elif rank_first > rank_second:
            self.parent[
                root_second
            ] = root_first

        else:
            self.parent[
                root_second
            ] = root_first

            self.rank[
                root_first
            ] += 1


def normalize_operator(
    operator: str,
) -> str:
    value = str(
        operator
    ).strip()

    if value not in {
        ">=",
        "<=",
        "==",
    }:
        raise ValueError(
            f"Unsupported threshold operator: {operator}"
        )

    return value


def condition_mask(
    series: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    operator = normalize_operator(
        operator
    )

    if operator == ">=":
        return numeric.ge(
            threshold
        )

    if operator == "<=":
        return numeric.le(
            threshold
        )

    return numeric.eq(
        threshold
    )


def concept_active_mask(
    dataframe: pd.DataFrame,
    member_1_feature: str,
    member_1_operator: str,
    member_1_threshold: float,
    member_2_feature: str,
    member_2_operator: str,
    member_2_threshold: float,
) -> np.ndarray:
    for feature_name in [
        member_1_feature,
        member_2_feature,
    ]:
        if feature_name not in dataframe.columns:
            raise KeyError(
                f"Feature missing from controlled view: {feature_name}"
            )

    first = pd.to_numeric(
        dataframe[
            member_1_feature
        ],
        errors="coerce",
    )

    second = pd.to_numeric(
        dataframe[
            member_2_feature
        ],
        errors="coerce",
    )

    available = (
        first.notna()
        & second.notna()
    )

    active = (
        available
        & condition_mask(
            first,
            member_1_operator,
            float(
                member_1_threshold
            ),
        )
        & condition_mask(
            second,
            member_2_operator,
            float(
                member_2_threshold
            ),
        )
    )

    return active.to_numpy(
        dtype=bool
    )


def hash_boolean_mask(
    mask: np.ndarray,
) -> str:
    packed = np.packbits(
        np.asarray(
            mask,
            dtype=np.uint8,
        )
    )

    return hashlib.sha256(
        packed.tobytes()
    ).hexdigest()


def member_condition_key(
    feature_name: str,
    operator: str,
    threshold: float,
) -> str:
    return (
        f"{str(feature_name)}"
        f"|{normalize_operator(operator)}"
        f"|{float(threshold):.12g}"
    )


def concept_member_set(
    member_1_key: str,
    member_2_key: str,
) -> frozenset[str]:
    return frozenset(
        [
            str(
                member_1_key
            ),
            str(
                member_2_key
            ),
        ]
    )


def jaccard_from_counts(
    intersection: int,
    first_count: int,
    second_count: int,
) -> float:
    union = (
        int(
            first_count
        )
        + int(
            second_count
        )
        - int(
            intersection
        )
    )

    if union <= 0:
        return 1.0

    return float(
        intersection
        / union
    )


def ranking_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    status_rank = {
        "STRONG_CONCEPT_CANDIDATE": 0,
        "CONCEPT_CANDIDATE": 1,
        "WEAK_CONCEPT_CANDIDATE": 2,
    }

    output[
        "_rank_status"
    ] = output[
        "concept_status"
    ].map(
        status_rank
    ).fillna(
        9
    )

    output[
        "_rank_q"
    ] = pd.to_numeric(
        output[
            "q_value"
        ],
        errors="coerce",
    ).fillna(
        np.inf
    )

    output[
        "_rank_incremental"
    ] = -pd.to_numeric(
        output[
            "incremental_lift_over_strongest_member"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_rank_lift"
    ] = -pd.to_numeric(
        output[
            "absolute_joint_lift"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_rank_sample"
    ] = -pd.to_numeric(
        output[
            "joint_active_sample"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    return output


def nominate_component_representatives(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    ranked = ranking_columns(
        dataframe
    )

    ranked = ranked.sort_values(
        [
            "target_name",
            "network_component_id",
            "_rank_status",
            "_rank_q",
            "_rank_incremental",
            "_rank_lift",
            "_rank_sample",
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
        ],
        kind="stable",
    )

    ranked[
        "network_component_rank"
    ] = (
        ranked.groupby(
            [
                "target_name",
                "network_component_id",
            ],
            sort=False,
        )
        .cumcount()
        + 1
    )

    ranked[
        "frozen_network_representative"
    ] = ranked[
        "network_component_rank"
    ].eq(
        1
    )

    return ranked.drop(
        columns=[
            "_rank_status",
            "_rank_q",
            "_rank_incremental",
            "_rank_lift",
            "_rank_sample",
        ],
        errors="ignore",
    )
