"""
Regression tests for the independent totals/scoring-shape target family.

Most of these tests use synthetic data only (no production files) so
they run in any environment. The integration tests at the bottom of
this file are the exception: they read the real, checked-in
``atlas_reference`` sample fixtures that mirror
``atlas.learning.factual_target_builder``'s actual output schema and
dtypes.
"""

from pathlib import Path

import pandas as pd
import pytest

from atlas.learning.factual_target_builder import (
    build_game_targets,
    build_team_game_targets,
)

from atlas.learning.totals_target_builder import (
    EXTREME_HIGH_SCORING_MIN_RUNS,
    FROZEN_SCORING_BUCKET_CONTRACT_PATH,
    FROZEN_SCORING_BUCKET_CONTRACT_VERSION,
    HIGH_SCORING_MIN_RUNS,
    LOW_SCORING_MAX_RUNS,
    RESERVED_REGULATION_EXTRA_INNING_COLUMNS,
    TOTAL_RUN_BUCKET_AVERAGE,
    TOTAL_RUN_BUCKET_EXTREME_HIGH,
    TOTAL_RUN_BUCKET_HIGH,
    TOTAL_RUN_BUCKET_LOW,
    WENT_EXTRA_INNINGS_COLUMN,
    build_total_runs_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

REAL_GAME_TARGETS_SAMPLE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "games"
    / "data__game_intelligence__factual_learning_targets__2024"
    "__factual_game_learning_targets.parquet.games.parquet"
)

REAL_TEAM_GAME_TARGETS_SAMPLE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "games"
    / "data__game_intelligence__factual_learning_targets__2024"
    "__factual_team_game_learning_targets.parquet.games.parquet"
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


def test_frozen_scoring_bucket_contract_governs_the_boundaries():
    import json

    with FROZEN_SCORING_BUCKET_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        contract = json.load(handle)

    assert contract["contract_version"] == FROZEN_SCORING_BUCKET_CONTRACT_VERSION
    assert contract["discovery_season"] == 2024
    assert contract["sample_size"] == 2428
    assert contract["percentile_method"]
    assert contract["percentile_values"] == {
        "p25": 5,
        "p50": 8,
        "p75": 11,
        "p90": 15,
    }
    assert contract["bucket_boundaries"] == {
        "low_scoring_max_runs": LOW_SCORING_MAX_RUNS,
        "high_scoring_min_runs": HIGH_SCORING_MIN_RUNS,
        "extreme_high_scoring_min_runs": EXTREME_HIGH_SCORING_MIN_RUNS,
    }
    assert contract["validation_isolation"]["2025_outcomes_used"] is False
    assert contract["validation_isolation"]["recomputed_at_runtime"] is False


def test_scoring_bucket_contract_version_column_is_stamped():
    totals = _build_totals()

    assert (
        totals["scoring_bucket_contract_version"]
        .eq(FROZEN_SCORING_BUCKET_CONTRACT_VERSION)
        .all()
    )


def test_one_team_explosion_vs_two_sided_high_scoring_distinguishes_12_1_from_7_6():
    completed = pd.DataFrame(
        {
            "game_pk": [10, 11],
            "game_date": pd.to_datetime(["2024-05-01", "2024-05-02"]),
            "atlas_season": [2024, 2024],
            "home_team": ["NYY", "BOS"],
            "away_team": ["BOS", "NYY"],
            "home_score": [12, 7],
            "away_score": [1, 6],
        }
    )

    game_targets = build_game_targets(completed)
    team_game_targets = build_team_game_targets(game_targets)
    totals = build_total_runs_targets(
        game_targets,
        team_game_targets,
    ).set_index("game_pk")

    # 12-1: one team's offensive explosion, a blowout high total.
    assert bool(totals.loc[10, "one_team_offensive_explosion"]) is True
    assert bool(totals.loc[10, "two_sided_high_scoring_game"]) is False
    assert bool(totals.loc[10, "blowout_high_total"]) is True
    assert bool(totals.loc[10, "competitive_high_total"]) is False
    assert totals.loc[10, "largest_team_run_total"] == 12

    # 7-6: two-sided, competitive high total -- not identical treatment.
    assert bool(totals.loc[11, "one_team_offensive_explosion"]) is False
    assert bool(totals.loc[11, "two_sided_high_scoring_game"]) is True
    assert bool(totals.loc[11, "blowout_high_total"]) is False
    assert bool(totals.loc[11, "competitive_high_total"]) is True
    assert totals.loc[11, "largest_team_run_total"] == 7


def test_home_and_away_scoring_share_sum_to_one_when_runs_are_scored():
    totals = _build_totals().set_index("game_pk")

    for game_pk in (1, 2, 3, 4):
        assert totals.loc[game_pk, "home_scoring_share"] + totals.loc[
            game_pk, "away_scoring_share"
        ] == pytest.approx(1.0)


def test_regulation_extra_inning_fields_are_reserved_not_fabricated():
    totals = _build_totals()

    for column in RESERVED_REGULATION_EXTRA_INNING_COLUMNS:
        assert column in totals.columns
        assert totals[column].isna().all()

    # went_extra_innings is reserved (all-null) when no game_outcomes
    # frame is supplied.
    assert WENT_EXTRA_INNINGS_COLUMN in totals.columns
    assert totals[WENT_EXTRA_INNINGS_COLUMN].isna().all()


def test_went_extra_innings_is_populated_when_game_outcomes_supplied():
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)

    game_outcomes = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "extra_innings": [False, True, False, False],
        }
    )

    totals = build_total_runs_targets(
        game_targets,
        team_game_targets,
        game_outcomes=game_outcomes,
    ).set_index("game_pk")

    assert bool(totals.loc[1, WENT_EXTRA_INNINGS_COLUMN]) is False
    assert bool(totals.loc[2, WENT_EXTRA_INNINGS_COLUMN]) is True

    # The remaining regulation/extra-innings fields stay reserved even
    # when game_outcomes is supplied -- no per-inning line score exists
    # to compute them accurately.
    for column in RESERVED_REGULATION_EXTRA_INNING_COLUMNS:
        assert totals[column].isna().all()


def test_missing_score_raises_instead_of_being_coerced_to_zero():
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)

    game_targets = game_targets.copy()
    game_targets.loc[0, "home_score"] = None

    with pytest.raises(ValueError):
        build_total_runs_targets(game_targets, team_game_targets)


def test_score_sum_mismatch_raises():
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)

    game_targets = game_targets.copy()
    game_targets.loc[0, "game_total_runs"] = 999

    with pytest.raises(AssertionError):
        build_total_runs_targets(game_targets, team_game_targets)


def test_non_final_game_state_raises():
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)

    game_targets = game_targets.copy()
    game_targets["game_state_category"] = "final"
    game_targets.loc[0, "game_state_category"] = "postponed"

    with pytest.raises(ValueError):
        build_total_runs_targets(game_targets, team_game_targets)


def test_final_game_state_passes():
    game_targets = build_game_targets(_completed_games())
    team_game_targets = build_team_game_targets(game_targets)

    game_targets = game_targets.copy()
    game_targets["game_state_category"] = "final"

    # Should not raise.
    build_total_runs_targets(game_targets, team_game_targets)


def test_no_projected_probability_or_market_columns_are_produced():
    totals = _build_totals()

    forbidden_prefixes_or_names = {
        "projected_home_runs",
        "projected_away_runs",
        "projected_total_runs",
        "over_probability",
        "under_probability",
        "over_under_selection",
    }

    assert forbidden_prefixes_or_names.isdisjoint(totals.columns)
    assert not totals["market_line_used"].any()


@pytest.mark.skipif(
    not REAL_GAME_TARGETS_SAMPLE.exists()
    or not REAL_TEAM_GAME_TARGETS_SAMPLE.exists(),
    reason="Real factual_target_builder output sample fixtures not found.",
)
def test_integration_against_real_factual_target_builder_output_schema():
    """
    Integration fixture: build totals targets directly from the real,
    checked-in ``atlas_reference`` sample parquet files that mirror
    ``atlas.learning.factual_target_builder``'s actual production
    output schema and dtypes (not synthetic data).
    """

    real_game_targets = pd.read_parquet(REAL_GAME_TARGETS_SAMPLE)
    real_team_game_targets = pd.read_parquet(REAL_TEAM_GAME_TARGETS_SAMPLE)

    totals = build_total_runs_targets(
        real_game_targets,
        real_team_game_targets,
    )

    assert len(totals) == len(real_game_targets)
    assert totals["game_pk"].is_unique

    assert (
        totals["actual_total_runs"]
        == totals["home_runs_scored"] + totals["away_runs_scored"]
    ).all()

    assert set(totals["total_run_bucket"].unique()) <= {
        TOTAL_RUN_BUCKET_LOW,
        TOTAL_RUN_BUCKET_AVERAGE,
        TOTAL_RUN_BUCKET_HIGH,
        TOTAL_RUN_BUCKET_EXTREME_HIGH,
    }

    assert not totals["market_line_used"].any()

    for column in RESERVED_REGULATION_EXTRA_INNING_COLUMNS:
        assert totals[column].isna().all()
