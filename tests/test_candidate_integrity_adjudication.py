import numpy as np
import pandas as pd

from atlas.learning.candidate_integrity_adjudication import (
    build_condition_mask_record,
    canonical_semantic_name,
    complement_trinary_state,
    direct_target_analogue,
    hash_state,
    nominate_group_representatives,
    source_classification,
    trinary_condition_state,
)


def test_trinary_condition_state():
    series = pd.Series([
        1.0,
        2.0,
        np.nan,
        4.0,
    ])

    state = trinary_condition_state(
        series=series,
        operator=">=",
        threshold=2.0,
    )

    assert state.tolist() == [
        1,
        2,
        0,
        2,
    ]


def test_complement_preserves_missing():
    state = np.array([
        1,
        2,
        0,
        2,
    ], dtype=np.uint8)

    complement = complement_trinary_state(
        state
    )

    assert complement.tolist() == [
        2,
        1,
        0,
        1,
    ]


def test_hash_is_deterministic():
    state = np.array([
        1,
        2,
        0,
    ], dtype=np.uint8)

    assert hash_state(
        state
    ) == hash_state(
        state.copy()
    )


def test_mask_record_matches_active_rows():
    dataframe = pd.DataFrame({
        "feature": [
            1.0,
            2.0,
            np.nan,
            4.0,
        ],
    })

    record = build_condition_mask_record(
        dataframe=dataframe,
        target_name="target_example",
        feature_name="feature",
        operator=">=",
        threshold=2.0,
    )

    assert record[
        "mask_active_rows"
    ] == 2

    assert record[
        "mask_inactive_rows"
    ] == 1

    assert record[
        "mask_missing_rows"
    ] == 1


def test_home_and_away_semantic_names_match():
    home = canonical_semantic_name(
        "home__bullpen__pitches_prior_3"
    )

    away = canonical_semantic_name(
        "away__bullpen__pitches_prior_3"
    )

    assert home == away


def test_direct_target_analogue_flag():
    flagged, tokens = direct_target_analogue(
        "target_team_win_by_2_plus",
        "identity__identity_edge__outcome__win_by_2_plus",
    )

    assert flagged
    assert "win_by_2_plus" in tokens


def test_source_classification():
    assert source_classification(
        "bullpen__pitches_prior_1"
    ) == "RAW_BULLPEN_PREGAME_FACT"

    assert source_classification(
        "identity__identity_edge__run_differential"
    ) == "DERIVED_IDENTITY_EDGE"


def test_one_representative_per_group():
    dataframe = pd.DataFrame({
        "target_name": [
            "target_a",
            "target_a",
        ],

        "undirected_mask_group_key": [
            "group_1",
            "group_1",
        ],

        "feature_name": [
            "feature_a",
            "feature_b",
        ],

        "condition_name": [
            "upper_quartile",
            "upper_quartile",
        ],

        "research_status": [
            "STRONG_DISCOVERY_CANDIDATE",
            "DISCOVERY_CANDIDATE",
        ],

        "q_value": [
            0.001,
            0.01,
        ],

        "absolute_lift": [
            0.10,
            0.08,
        ],

        "active_sample": [
            100,
            100,
        ],

        "direct_target_analogue": [
            False,
            False,
        ],

        "source_classification": [
            "RAW_BULLPEN_PREGAME_FACT",
            "RAW_BULLPEN_PREGAME_FACT",
        ],
    })

    nominated = nominate_group_representatives(
        dataframe
    )

    assert nominated[
        "nominated_representative"
    ].sum() == 1

    assert nominated.loc[
        nominated[
            "nominated_representative"
        ],
        "feature_name",
    ].iloc[
        0
    ] == "feature_a"


def test_direct_analogue_does_not_beat_non_analogue():
    dataframe = pd.DataFrame({
        "target_name": [
            "target_a",
            "target_a",
        ],

        "undirected_mask_group_key": [
            "group_1",
            "group_1",
        ],

        "feature_name": [
            "analogue_feature",
            "raw_feature",
        ],

        "condition_name": [
            "upper_quartile",
            "upper_quartile",
        ],

        "research_status": [
            "STRONG_DISCOVERY_CANDIDATE",
            "DISCOVERY_CANDIDATE",
        ],

        "q_value": [
            0.0001,
            0.02,
        ],

        "absolute_lift": [
            0.20,
            0.06,
        ],

        "active_sample": [
            100,
            100,
        ],

        "direct_target_analogue": [
            True,
            False,
        ],

        "source_classification": [
            "DERIVED_IDENTITY_EDGE",
            "RAW_BULLPEN_PREGAME_FACT",
        ],
    })

    nominated = nominate_group_representatives(
        dataframe
    )

    representative = nominated.loc[
        nominated[
            "nominated_representative"
        ],
        "feature_name",
    ].iloc[
        0
    ]

    assert representative == "raw_feature"
