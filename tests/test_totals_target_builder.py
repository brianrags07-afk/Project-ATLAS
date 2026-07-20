"""
Regression tests for the independent totals/scoring-shape target family.

These tests use synthetic data only (no production files) so they run
in any environment.
"""

import pandas as pd
import pytest

from atlas.learning.factual_target_builder import (
    build_game_targets,
    build_team_game_targets,
)

from atlas.learning.totals_target_builder import (
    EXTREME_HIGH_SCORING_MIN_RUNS,
    HIGH_SCORING_MIN_RUNS,
    LOW_SCORING_MAX_RUNS,
    TOTAL_RUN_BUCKET_AVERAGE,
    TOTAL_RUN_BUCKET_EXTREME_HIGH,
    TOTAL_RUN_BUCKET_HIGH,
    TOTAL_RUN_BUCKET_LOW,
    build_total_runs_targets,
)


def _completed_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "game_date": pd.to_datetime(
                ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]
            ),
            "atlas_season": [2024, 2024, 2024, 2024],
            "home_team": ["NYY", "BOS", "LAD", "SD"],
            "away_team": ["BOS", "NYY", "SD", "LAD"],
            # totals: 3 (low), 8 (average), 12 (high), 18 (extreme high)
            "home_score": [2, 5, 7, 10],
            "away_score": [1, 3, 5, 8],
        }
    )


def _build_totals() -> pd.DataFrame:
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)
    return build_total_runs_targets(game_targets, team_game_targets)


def test_total_run_bucket_boundaries_match_2024_derived_thresholds():
    assert LOW_SCORING_MAX_RUNS == 5
    assert HIGH_SCORING_MIN_RUNS == 12
    assert EXTREME_HIGH_SCORING_MIN_RUNS == 15


def test_actual_total_runs_matches_home_plus_away():
    totals = _build_totals()

    assert list(
        totals["actual_total_runs"]
    ) == list(
        totals["home_runs_scored"] + totals["away_runs_scored"]
    )


def test_low_average_high_extreme_high_buckets():
    totals = _build_totals().set_index("game_pk")

    assert totals.loc[1, "total_run_bucket"] == TOTAL_RUN_BUCKET_LOW
    assert bool(totals.loc[1, "low_scoring_game"]) is True
    assert bool(totals.loc[1, "high_scoring_game"]) is False

    assert totals.loc[2, "total_run_bucket"] == TOTAL_RUN_BUCKET_AVERAGE
    assert bool(totals.loc[2, "low_scoring_game"]) is False
    assert bool(totals.loc[2, "high_scoring_game"]) is False

    assert totals.loc[3, "total_run_bucket"] == TOTAL_RUN_BUCKET_HIGH
    assert bool(totals.loc[3, "high_scoring_game"]) is True
    assert bool(totals.loc[3, "extreme_high_scoring_game"]) is False

    assert totals.loc[4, "total_run_bucket"] == TOTAL_RUN_BUCKET_EXTREME_HIGH
    assert bool(totals.loc[4, "high_scoring_game"]) is True
    assert bool(totals.loc[4, "extreme_high_scoring_game"]) is True


def test_per_side_scoring_shape_matches_team_game_targets():
    totals = _build_totals().set_index("game_pk")

    # Game 3: home scored 7 (not <=3, not ==4, is >=5), away scored 5 (>=5)
    assert bool(totals.loc[3, "home_team_scored_3_or_less"]) is False
    assert bool(totals.loc[3, "home_team_scored_exactly_4"]) is False
    assert bool(totals.loc[3, "home_team_scored_5_plus"]) is True
    assert bool(totals.loc[3, "away_team_scored_5_plus"]) is True

    # Game 1: home scored 2 (<=3), away scored 1 (<=3)
    assert bool(totals.loc[1, "home_team_scored_3_or_less"]) is True
    assert bool(totals.loc[1, "away_team_scored_3_or_less"]) is True


def test_totals_family_is_structurally_independent_of_moneyline_and_run_margin():
    totals = _build_totals()

    forbidden_columns = {
        "target_team_win",
        "target_team_win_by_2_plus",
        "target_team_loss",
        "run_margin",
        "home_margin",
        "away_margin",
        "margin_2_plus",
        "margin_4_plus",
        "margin_6_plus",
    }

    assert forbidden_columns.isdisjoint(totals.columns)
    assert totals["moneyline_independent"].all()
    assert totals["run_margin_independent"].all()


def test_market_line_never_used_by_totals_family():
    totals = _build_totals()

    assert not totals["market_line_used"].any()


def test_inputs_are_not_mutated():
    completed = _completed_games()
    game_targets = build_game_targets(completed)
    game_targets_columns_before = list(game_targets.columns)
    team_game_targets = build_team_game_targets(game_targets)
    team_game_targets_columns_before = list(team_game_targets.columns)

    build_total_runs_targets(game_targets, team_game_targets)

    assert list(game_targets.columns) == game_targets_columns_before
    assert list(team_game_targets.columns) == team_game_targets_columns_before


def test_missing_required_column_raises():
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)

    with pytest.raises(KeyError):
        build_total_runs_targets(
            game_targets.drop(columns=["game_total_runs"]),
            team_game_targets,
        )
