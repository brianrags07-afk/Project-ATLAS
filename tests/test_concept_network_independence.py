import numpy as np
import pandas as pd

from atlas.learning.concept_network_independence import (
    UnionFind,
    concept_active_mask,
    concept_member_set,
    hash_boolean_mask,
    jaccard_from_counts,
    member_condition_key,
    nominate_component_representatives,
)


def test_union_find_connects_values():
    union_find = UnionFind(
        [
            "a",
            "b",
            "c",
        ]
    )

    union_find.union(
        "a",
        "b",
    )

    assert union_find.find(
        "a"
    ) == union_find.find(
        "b"
    )

    assert union_find.find(
        "a"
    ) != union_find.find(
        "c"
    )


def test_member_condition_key_is_deterministic():
    first = member_condition_key(
        "feature_a",
        ">=",
        1.25,
    )

    second = member_condition_key(
        "feature_a",
        ">=",
        1.25,
    )

    assert first == second


def test_member_set_is_order_independent():
    first = concept_member_set(
        "a",
        "b",
    )

    second = concept_member_set(
        "b",
        "a",
    )

    assert first == second


def test_jaccard_from_counts():
    result = jaccard_from_counts(
        intersection=8,
        first_count=10,
        second_count=10,
    )

    assert result == 8 / 12


def test_empty_jaccard_is_one():
    result = jaccard_from_counts(
        intersection=0,
        first_count=0,
        second_count=0,
    )

    assert result == 1.0


def test_concept_active_mask():
    dataframe = pd.DataFrame({
        "first": [
            1.0,
            3.0,
            4.0,
            np.nan,
        ],
        "second": [
            5.0,
            1.0,
            6.0,
            7.0,
        ],
    })

    mask = concept_active_mask(
        dataframe=dataframe,
        member_1_feature="first",
        member_1_operator=">=",
        member_1_threshold=3.0,
        member_2_feature="second",
        member_2_operator=">=",
        member_2_threshold=5.0,
    )

    assert mask.tolist() == [
        False,
        False,
        True,
        False,
    ]


def test_boolean_hash_is_deterministic():
    mask = np.array(
        [
            True,
            False,
            True,
        ]
    )

    assert hash_boolean_mask(
        mask
    ) == hash_boolean_mask(
        mask.copy()
    )


def test_component_nomination_prefers_strong_status():
    dataframe = pd.DataFrame({
        "target_name": [
            "target_a",
            "target_a",
        ],
        "network_component_id": [
            "component_a",
            "component_a",
        ],
        "concept_id": [
            "candidate",
            "strong",
        ],
        "concept_status": [
            "CONCEPT_CANDIDATE",
            "STRONG_CONCEPT_CANDIDATE",
        ],
        "q_value": [
            0.001,
            0.01,
        ],
        "incremental_lift_over_strongest_member": [
            0.10,
            0.05,
        ],
        "absolute_joint_lift": [
            0.20,
            0.15,
        ],
        "joint_active_sample": [
            500,
            400,
        ],
    })

    ranked = nominate_component_representatives(
        dataframe
    )

    representative = ranked.loc[
        ranked[
            "frozen_network_representative"
        ],
        "concept_id",
    ].iloc[
        0
    ]

    assert representative == "strong"


def test_one_representative_per_component():
    dataframe = pd.DataFrame({
        "target_name": [
            "target_a",
            "target_a",
            "target_a",
        ],
        "network_component_id": [
            "component_a",
            "component_a",
            "component_b",
        ],
        "concept_id": [
            "a",
            "b",
            "c",
        ],
        "concept_status": [
            "STRONG_CONCEPT_CANDIDATE",
            "CONCEPT_CANDIDATE",
            "WEAK_CONCEPT_CANDIDATE",
        ],
        "q_value": [
            0.01,
            0.02,
            0.03,
        ],
        "incremental_lift_over_strongest_member": [
            0.05,
            0.04,
            0.03,
        ],
        "absolute_joint_lift": [
            0.10,
            0.09,
            0.08,
        ],
        "joint_active_sample": [
            100,
            90,
            80,
        ],
    })

    ranked = nominate_component_representatives(
        dataframe
    )

    representatives = (
        ranked.groupby(
            "network_component_id"
        )[
            "frozen_network_representative"
        ]
        .sum()
    )

    assert representatives.eq(
        1
    ).all()
