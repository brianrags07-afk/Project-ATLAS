from pathlib import Path

import pandas as pd

from atlas.pregame.clean_starter_lineup_pregame import (
    ADAPTER_VERSION,
    classify_column,
    clean_pregame_artifact,
)


ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

BATTER_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/pregame/snapshots/batter_pregame_snapshots.parquet'
)

PITCHER_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/pregame/snapshots/pitcher_pregame_snapshots.parquet'
)

LINEUP_STARTER_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/pregame/interactions/lineup_starter_inputs.parquet'
)


def test_adapter_version():
    assert ADAPTER_VERSION == "1.0.0"


def test_canonical_sources_exist():
    assert BATTER_PATH.exists()
    assert PITCHER_PATH.exists()
    assert LINEUP_STARTER_PATH.exists()


def test_explicit_targets_are_blocked():
    for column in [
        "actual_runs_allowed",
        "game_result",
        "target_win_by_2_plus",
        "final_score",
        "total_runs",
    ]:
        action, _ = classify_column(
            column
        )

        assert action == "BLOCK_POSTGAME"


def test_historical_performance_is_retained():
    for column in [
        "prior_runs_allowed",
        "rolling_strikeouts_rate",
        "career_walk_rate",
        "season_to_date_hits_allowed",
    ]:
        action, _ = classify_column(
            column
        )

        assert action == "KEEP_HISTORICAL_FACT"


def test_handcrafted_fields_are_blocked():
    for column in [
        "starter_score",
        "lineup_matchup_score",
        "confidence_score",
        "prediction_probability",
    ]:
        action, _ = classify_column(
            column
        )

        assert action == "BLOCK_HANDCRAFTED"


def test_clean_batter_snapshot():
    source = pd.read_parquet(
        BATTER_PATH
    )

    clean, registry = clean_pregame_artifact(
        source
    )

    assert not clean.empty
    assert not registry.empty
    assert clean["strict_backtest_safe"].all()
    assert not clean["future_games_used"].any()
    assert not clean["same_date_games_used"].any()


def test_clean_pitcher_snapshot():
    source = pd.read_parquet(
        PITCHER_PATH
    )

    clean, registry = clean_pregame_artifact(
        source
    )

    assert not clean.empty

    blocked_columns = set(
        registry.loc[
            registry[
                "governance_action"
            ].isin(
                [
                    "BLOCK_POSTGAME",
                    "BLOCK_HANDCRAFTED",
                    "MANUAL_REVIEW",
                ]
            ),
            "column",
        ]
    )

    assert blocked_columns.isdisjoint(
        clean.columns
    )


def test_clean_lineup_starter_input():
    source = pd.read_parquet(
        LINEUP_STARTER_PATH
    )

    clean, registry = clean_pregame_artifact(
        source
    )

    assert not clean.empty

    blocked_columns = set(
        registry.loc[
            registry[
                "governance_action"
            ].isin(
                [
                    "BLOCK_POSTGAME",
                    "BLOCK_HANDCRAFTED",
                    "MANUAL_REVIEW",
                ]
            ),
            "column",
        ]
    )

    assert blocked_columns.isdisjoint(
        clean.columns
    )


def test_clean_outputs_reach_july_3_2026():
    for path in [
        BATTER_PATH,
        PITCHER_PATH,
        LINEUP_STARTER_PATH,
    ]:
        source = pd.read_parquet(
            path
        )

        clean, _ = clean_pregame_artifact(
            source
        )

        latest_date = pd.to_datetime(
            clean["game_date"]
        ).max()

        assert latest_date >= pd.Timestamp(
            "2026-07-03"
        )


def test_saved_clean_artifacts_exist():
    base = (
        ROOT
        / "data"
        / "game_intelligence"
        / "clean_starter_lineup_pregame"
    )

    expected = [
        (
            base
            / "batter_pregame_snapshots"
            / "2026"
            / "batter_pregame_snapshots.parquet"
        ),
        (
            base
            / "pitcher_pregame_snapshots"
            / "2026"
            / "pitcher_pregame_snapshots.parquet"
        ),
        (
            base
            / "lineup_starter_inputs"
            / "2026"
            / "lineup_starter_inputs.parquet"
        ),
    ]

    for path in expected:
        assert path.exists(), path
