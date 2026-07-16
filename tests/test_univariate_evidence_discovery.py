import numpy as np
import pandas as pd

from atlas.learning.univariate_evidence_discovery import (
    benjamini_hochberg,
    discover_target,
    two_proportion_p_value,
)


def sample_view():
    rows = 120

    return pd.DataFrame({
        "game_pk":
            np.arange(
                rows
            ),

        "game_date":
            pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="D",
            ),

        "atlas_season":
            2024,

        "team":
            "AAA",

        "opponent":
            "BBB",

        "home_away":
            "HOME",

        "identity__example_numeric":
            np.arange(
                rows,
                dtype=float,
            ),

        "bullpen__example_boolean":
            np.arange(
                rows
            ) % 2 == 0,

        "target_label":
            np.arange(
                rows
            ) >= 80,

        "target_name":
            "target_example",

        "discovery_grain":
            "TEAM_GAME",

        "strict_backtest_safe":
            True,

        "prediction_created":
            False,

        "weight_assigned":
            False,
    })


def test_two_proportion_p_value_range():
    p_value = two_proportion_p_value(
        active_successes=30,
        active_sample=50,
        inactive_successes=10,
        inactive_sample=50,
    )

    assert 0.0 <= p_value <= 1.0


def test_bh_q_values_are_valid():
    p_values = pd.Series([
        0.001,
        0.01,
        0.03,
        0.50,
    ])

    q_values = benjamini_hochberg(
        p_values
    )

    assert q_values.between(
        0.0,
        1.0,
        inclusive="both",
    ).all()


def test_discovery_creates_numeric_conditions():
    registry, audit = discover_target(
        dataframe=sample_view(),
        target_name="target_example",
        discovery_grain="TEAM_GAME",
    )

    assert not registry.empty

    assert audit[
        "eligible"
    ].any()

    assert set(
        registry[
            "condition_name"
        ]
    ).intersection({
        "lower_quartile",
        "upper_quartile",
        "equals_high_value",
    })


def test_discovery_does_not_assign_weights():
    registry, _ = discover_target(
        dataframe=sample_view(),
        target_name="target_example",
        discovery_grain="TEAM_GAME",
    )

    assert not registry[
        "prediction_weight_assigned"
    ].any()

    assert not registry[
        "prediction_created"
    ].any()


def test_target_and_context_are_not_features():
    registry, audit = discover_target(
        dataframe=sample_view(),
        target_name="target_example",
        discovery_grain="TEAM_GAME",
    )

    tested = set(
        registry[
            "feature_name"
        ]
    )

    assert "target_label" not in tested
    assert "game_pk" not in tested
    assert "game_date" not in tested


def test_expected_strong_relationship_is_detected():
    registry, _ = discover_target(
        dataframe=sample_view(),
        target_name="target_example",
        discovery_grain="TEAM_GAME",
    )

    numeric = registry[
        registry[
            "feature_name"
        ].eq(
            "identity__example_numeric"
        )
    ]

    assert numeric[
        "absolute_lift"
    ].max() > 0
