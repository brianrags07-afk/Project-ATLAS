import numpy as np
import pandas as pd

from atlas.learning.concept_integrity_adjudication import (
    broad_evidence_domain,
    broad_domain_pair_key,
    complement_trinary_state,
    concept_structural_status,
    hash_state,
    lineage_pair_key,
    nominate_joint_mask_representatives,
    trinary_joint_state,
    underlying_metric_root,
    undirected_joint_mask_key,
)


def test_broad_evidence_domains():
    assert broad_evidence_domain(
        "DERIVED_IDENTITY_EDGE"
    ) == "IDENTITY"

    assert broad_evidence_domain(
        "RAW_BULLPEN_PREGAME_FACT"
    ) == "BULLPEN"

    assert broad_evidence_domain(
        "LINEUP_STARTER_PREGAME_FACT"
    ) == "LINEUP_STARTER"


def test_team_and_opponent_identity_share_metric_root():
    team_feature = (
        "identity__team_identity__lead__maximum_lead"
    )

    opponent_feature = (
        "identity__opponent_identity__lead__maximum_lead"
    )

    assert underlying_metric_root(
        team_feature
    ) == underlying_metric_root(
        opponent_feature
    )


def test_edge_and_team_identity_share_underlying_metric():
    edge_feature = (
        "identity__identity_edge__run_differential"
    )

    team_feature = (
        "identity__team_identity__run_differential"
    )

    assert underlying_metric_root(
        edge_feature
    ) == underlying_metric_root(
        team_feature
    )


def test_distinct_baseball_metrics_remain_distinct():
    first = (
        "identity__identity_edge__run_differential"
    )

    second = (
        "identity__identity_edge__lead__maximum_lead"
    )

    assert underlying_metric_root(
        first
    ) != underlying_metric_root(
        second
    )


def test_lineage_pair_is_order_independent():
    first = lineage_pair_key(
        "identity::run_differential",
        "bullpen::walk_rate",
    )

    second = lineage_pair_key(
        "bullpen::walk_rate",
        "identity::run_differential",
    )

    assert first == second


def test_broad_domain_pair_is_order_independent():
    first = broad_domain_pair_key(
        "IDENTITY",
        "BULLPEN",
    )

    second = broad_domain_pair_key(
        "BULLPEN",
        "IDENTITY",
    )

    assert first == second


def test_trinary_joint_state():
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

    state = trinary_joint_state(
        dataframe=dataframe,
        member_1_feature="first",
        member_1_operator=">=",
        member_1_threshold=3.0,
        member_2_feature="second",
        member_2_operator=">=",
        member_2_threshold=5.0,
    )

    assert state.tolist() == [
        1,
        1,
        2,
        0,
    ]


def test_complement_preserves_missing():
    state = np.array(
        [
            1,
            2,
            0,
        ],
        dtype=np.uint8,
    )

    complement = complement_trinary_state(
        state
    )

    assert complement.tolist() == [
        2,
        1,
        0,
    ]


def test_hash_is_deterministic():
    state = np.array(
        [
            1,
            2,
            0,
        ],
        dtype=np.uint8,
    )

    assert hash_state(
        state
    ) == hash_state(
        state.copy()
    )


def test_undirected_mask_key_is_order_independent():
    assert undirected_joint_mask_key(
        "aaa",
        "bbb",
    ) == undirected_joint_mask_key(
        "bbb",
        "aaa",
    )


def test_structural_status_blocks_same_metric():
    status = concept_structural_status(
        same_underlying_metric=True,
        same_broad_domain=False,
        mirrored_identity_pair=False,
        exact_joint_duplicate=False,
        inverse_joint_duplicate=False,
        nominated_representative=True,
    )

    assert status == "BLOCK_SHARED_UNDERLYING_METRIC"


def test_structural_status_blocks_same_domain():
    status = concept_structural_status(
        same_underlying_metric=False,
        same_broad_domain=True,
        mirrored_identity_pair=False,
        exact_joint_duplicate=False,
        inverse_joint_duplicate=False,
        nominated_representative=True,
    )

    assert status == "BLOCK_SINGLE_BROAD_DOMAIN"


def test_nomination_prefers_structurally_independent_row():
    dataframe = pd.DataFrame({
        "target_name": [
            "target_a",
            "target_a",
        ],
        "undirected_joint_mask_key": [
            "group_a",
            "group_a",
        ],
        "concept_id": [
            "blocked_concept",
            "independent_concept",
        ],
        "same_underlying_metric": [
            True,
            False,
        ],
        "same_broad_domain": [
            True,
            False,
        ],
        "mirrored_identity_pair": [
            True,
            False,
        ],
        "concept_status": [
            "STRONG_CONCEPT_CANDIDATE",
            "CONCEPT_CANDIDATE",
        ],
        "q_value": [
            0.0001,
            0.01,
        ],
        "incremental_lift_over_strongest_member": [
            0.10,
            0.03,
        ],
        "absolute_joint_lift": [
            0.20,
            0.08,
        ],
        "joint_active_sample": [
            500,
            300,
        ],
    })

    ranked = nominate_joint_mask_representatives(
        dataframe
    )

    representative = ranked.loc[
        ranked[
            "nominated_joint_mask_representative"
        ],
        "concept_id",
    ].iloc[
        0
    ]

    assert representative == "independent_concept"
