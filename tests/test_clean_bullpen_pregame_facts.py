from pathlib import Path
import re

import pandas as pd

from atlas.pregame.clean_bullpen_pregame_facts import (
    ADAPTER_VERSION,
    BLOCKED_COLUMN_PATTERNS,
    LEAKAGE_COLUMN_PATTERNS,
    build_clean_bullpen_pregame_facts,
)


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

CANONICAL_SOURCE = Path(
    '/content/drive/MyDrive/Project_Atlas/data/pregame/bullpen/bullpen_pregame_state.parquet'
)

OUTPUT_BASE = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "clean_bullpen_pregame_facts"
)


def matches_any(column_name, patterns):
    return any(
        re.search(
            pattern,
            str(column_name).lower(),
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def test_adapter_version():
    assert ADAPTER_VERSION == "1.0.0"


def test_canonical_source_exists():
    assert CANONICAL_SOURCE.exists()


def test_clean_builder_blocks_handcrafted_columns():
    source = pd.read_parquet(CANONICAL_SOURCE)

    clean, registry = build_clean_bullpen_pregame_facts(
        source
    )

    assert not clean.empty
    assert not registry.empty

    blocked = [
        column
        for column in clean.columns
        if matches_any(
            column,
            BLOCKED_COLUMN_PATTERNS,
        )
    ]

    assert blocked == []


def test_clean_builder_blocks_postgame_columns():
    source = pd.read_parquet(CANONICAL_SOURCE)

    clean, _ = build_clean_bullpen_pregame_facts(
        source
    )

    leakage = [
        column
        for column in clean.columns
        if matches_any(
            column,
            LEAKAGE_COLUMN_PATTERNS,
        )
    ]

    assert leakage == []


def test_clean_builder_has_required_keys():
    source = pd.read_parquet(CANONICAL_SOURCE)

    clean, _ = build_clean_bullpen_pregame_facts(
        source
    )

    required = {
        "game_pk",
        "game_date",
        "team",
        "atlas_season",
    }

    assert required.issubset(clean.columns)


def test_clean_builder_has_no_duplicate_team_games():
    source = pd.read_parquet(CANONICAL_SOURCE)

    clean, _ = build_clean_bullpen_pregame_facts(
        source
    )

    assert (
        clean.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
        == 0
    )


def test_clean_builder_sets_chronology_controls():
    source = pd.read_parquet(CANONICAL_SOURCE)

    clean, _ = build_clean_bullpen_pregame_facts(
        source
    )

    assert clean["strict_backtest_safe"].all()
    assert not clean["same_date_games_used"].any()
    assert not clean["future_games_used"].any()
    assert not clean["handcrafted_scores_included"].any()


def test_saved_season_artifacts_exist():
    season_directories = [
        path
        for path in OUTPUT_BASE.iterdir()
        if path.is_dir()
        and path.name.isdigit()
    ]

    assert season_directories

    for season_dir in season_directories:
        facts_path = (
            season_dir
            / "clean_bullpen_pregame_facts.parquet"
        )

        metadata_path = (
            season_dir
            / "clean_bullpen_pregame_fact_metadata.json"
        )

        assert facts_path.exists()
        assert metadata_path.exists()


def test_saved_artifacts_have_no_blocked_columns():
    season_directories = [
        path
        for path in OUTPUT_BASE.iterdir()
        if path.is_dir()
        and path.name.isdigit()
    ]

    for season_dir in season_directories:
        frame = pd.read_parquet(
            season_dir
            / "clean_bullpen_pregame_facts.parquet"
        )

        blocked = [
            column
            for column in frame.columns
            if matches_any(
                column,
                BLOCKED_COLUMN_PATTERNS,
            )
        ]

        leakage = [
            column
            for column in frame.columns
            if matches_any(
                column,
                LEAKAGE_COLUMN_PATTERNS,
            )
        ]

        assert blocked == []
        assert leakage == []
