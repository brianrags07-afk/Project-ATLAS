import numpy as np
import pandas as pd

from atlas.learning.feature_semantic_governance import (
    base_metric_root,
    classify_feature,
    combine_member_actions,
    infer_value_profile,
    is_identifier_feature,
    nominate_transformation_family_representatives,
    transformation_family_root,
)


def numeric_profile():
    return {
        "numeric_compatible": True,
        "integer_like": False,
        "binary_like": False,
        "unique_values": 100,
    }


def test_player_id_is_identifier():
    assert is_identifier_feature(
        "lineup_starter__batting_order_3_player_id"
    )


def test_pitcher_id_is_identifier():
    assert is_identifier_feature(
        "lineup_starter__starter_pitcher_id"
    )


def test_identifier_is_blocked():
    classification, action, _ = classify_feature(
        feature_name="lineup_starter__batting_order_3_player_id",
        dtype_name="int64",
        value_profile={
            "numeric_compatible": True,
            "integer_like": True,
            "binary_like": False,
            "unique_values": 500,
        },
    )

    assert classification == "IDENTIFIER_NOT_MEASUREMENT"
    assert action == "BLOCK_IDENTIFIER_THRESHOLD"


def test_rate_is_retained():
    classification, action, _ = classify_feature(
        feature_name="bullpen__bullpen_walk_per_pitch_season_prior_mean",
        dtype_name="float64",
        value_profile=numeric_profile(),
    )

    assert classification == "VALID_RATE_BASEBALL_FACT"
    assert action == "KEEP_SEMANTICALLY_VALID"


def test_exposure_is_reviewed():
    classification, action, _ = classify_feature(
        feature_name="lineup_starter__starter_career_prior_plate_appearances",
        dtype_name="int64",
        value_profile={
            "numeric_compatible": True,
            "integer_like": True,
            "binary_like": False,
            "unique_values": 100,
        },
    )

    assert classification == "SAMPLE_SIZE_OR_EXPOSURE_PROXY"
    assert action == "REVIEW_EXPOSURE_PROXY"


def test_target_analogue_is_blocked():
    classification, action, _ = classify_feature(
        feature_name="identity__team_identity__outcome__win_by_2_plus",
        dtype_name="float64",
        value_profile=numeric_profile(),
    )

    assert classification == "POTENTIAL_TARGET_ANALOGUE"
    assert action == "BLOCK_TARGET_ANALOGUE"


def test_profile_handles_numeric_series():
    profile = infer_value_profile(
        pd.Series(
            [
                1.0,
                2.0,
                np.nan,
            ]
        )
    )

    assert profile[
        "numeric_compatible"
    ]

    assert profile[
        "missing_count"
    ] == 1


def test_transformation_family_collapses_career_and_season():
    career = transformation_family_root(
        "lineup_starter__lineup_career_prior_home_run_rate_per_pa_mean"
    )

    season = transformation_family_root(
        "lineup_starter__lineup_season_prior_home_run_rate_per_pa_mean"
    )

    assert career == season


def test_base_metric_collapses_aggregation_suffix():
    first = base_metric_root(
        "lineup_starter__lineup_career_prior_ball_pct_mean"
    )

    second = base_metric_root(
        "lineup_starter__lineup_career_prior_ball_pct_max"
    )

    assert first == second


def test_blocked_member_blocks_concept():
    status, _ = combine_member_actions(
        "KEEP_SEMANTICALLY_VALID",
        "BLOCK_IDENTIFIER_THRESHOLD",
    )

    assert status == "BLOCKED_INVALID_MEMBER"


def test_exposure_member_requires_review():
    status, _ = combine_member_actions(
        "KEEP_SEMANTICALLY_VALID",
        "REVIEW_EXPOSURE_PROXY",
    )

    assert status == "REVIEW_REQUIRED_MEMBER_SEMANTICS"


def test_two_valid_members_pass():
    status, _ = combine_member_actions(
        "KEEP_SEMANTICALLY_VALID",
        "KEEP_SEMANTICALLY_VALID",
    )

    assert status == "SEMANTICALLY_VALID_FREEZE_CANDIDATE"


def test_family_nomination_creates_one_representative():
    dataframe = pd.DataFrame({
        "concept_id": [
            "a",
            "b",
        ],
        "target_name": [
            "target_a",
            "target_a",
        ],
        "semantic_family_pair_key": [
            "family_a",
            "family_a",
        ],
        "concept_status": [
            "STRONG_CONCEPT_CANDIDATE",
            "CONCEPT_CANDIDATE",
        ],
        "q_value": [
            0.01,
            0.001,
        ],
        "incremental_lift_over_strongest_member": [
            0.04,
            0.08,
        ],
        "absolute_joint_lift": [
            0.10,
            0.15,
        ],
        "joint_active_sample": [
            200,
            300,
        ],
    })

    result = nominate_transformation_family_representatives(
        dataframe
    )

    assert result[
        "semantic_family_representative"
    ].sum() == 1

    representative = result.loc[
        result[
            "semantic_family_representative"
        ],
        "concept_id",
    ].iloc[
        0
    ]

    assert representative == "a"



def test_recent_bullpen_usage_is_valid_workload():
    classification, action, _ = classify_feature(
        feature_name="bullpen__bullpen_games_used_prior_3_dates",
        dtype_name="float64",
        value_profile={
            "numeric_compatible": True,
            "integer_like": True,
            "binary_like": False,
            "unique_values": 4,
        },
    )

    assert classification == "VALID_RECENT_WORKLOAD_FACT"
    assert action == "KEEP_SEMANTICALLY_VALID"


def test_historical_games_remain_exposure_proxy():
    classification, action, _ = classify_feature(
        feature_name="starter_career_prior_games",
        dtype_name="int64",
        value_profile={
            "numeric_compatible": True,
            "integer_like": True,
            "binary_like": False,
            "unique_values": 100,
        },
    )

    assert classification == "SAMPLE_SIZE_OR_EXPOSURE_PROXY"
    assert action == "REVIEW_EXPOSURE_PROXY"
