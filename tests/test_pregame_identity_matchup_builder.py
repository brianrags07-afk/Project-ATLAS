import json
from pathlib import Path

import pandas as pd
import pytest

from atlas.game_intelligence.pregame_identity_matchup_builder import (
    ENGINE_VERSION,
    assert_reproduces_reference_matchups,
    build_pregame_identity_matchups,
    build_pregame_identity_matchups_metadata,
    phase_2e_identity_matchup_paths,
    save_pregame_identity_matchups,
)


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

TIMELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "pregame_team_identities"
    / "2024"
    / "pregame_team_identity_timeline.parquet"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "pregame_identity_registry"
    / "2024"
    / "pregame_identity_source_registry.csv"
)

REFERENCE_MATCHUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "pregame_identity_matchups"
    / "2024"
    / "pregame_identity_matchups.parquet"
)


def _sample_timeline() -> pd.DataFrame:
    return pd.DataFrame({
        "game_pk": [100, 100, 101, 101],
        "game_date": [
            "2024-05-01",
            "2024-05-01",
            "2024-05-02",
            "2024-05-02",
        ],
        "atlas_season": [2024, 2024, 2024, 2024],
        "team": ["NYY", "BOS", "NYY", "BOS"],
        "opponent": ["BOS", "NYY", "BOS", "NYY"],
        "home_away": ["HOME", "AWAY", "AWAY", "HOME"],
        "identity_source_game_pk": [99, 98, 100, 100],
        "identity_source_game_date": [
            "2024-04-30",
            "2024-04-30",
            "2024-05-01",
            "2024-05-01",
        ],
        "strict_backtest_safe": [True, True, True, True],
        "same_date_games_used": [False, False, False, False],
        "future_games_used": [False, False, False, False],
        "rolling_runs": [5.0, 3.0, 4.5, 3.5],
        "prior_whip": [1.05, 1.20, 1.10, 1.15],
    })


def _sample_registry() -> pd.DataFrame:
    return pd.DataFrame({
        "identity_feature_name": [
            "rolling_runs",
            "prior_whip",
        ],
        "source_column": [
            "rolling_runs",
            "prior_whip",
        ],
    })


def test_engine_version():
    assert ENGINE_VERSION == "1.0.0"


def test_matchup_builder_creates_mirrored_edges():
    matchups, mirror_audit = build_pregame_identity_matchups(
        timeline=_sample_timeline(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    assert mirror_audit["audit_pass"].all()

    game_100 = matchups[
        matchups["game_pk"].eq(100)
    ].sort_values("team", kind="stable")

    nyy_edge = game_100.loc[
        game_100["team"].eq("NYY"),
        "identity_edge__rolling_runs",
    ].iloc[0]

    bos_edge = game_100.loc[
        game_100["team"].eq("BOS"),
        "identity_edge__rolling_runs",
    ].iloc[0]

    assert nyy_edge == 2.0
    assert bos_edge == -2.0


def test_matchup_paths_and_save(tmp_path):
    matchups, mirror_audit = build_pregame_identity_matchups(
        timeline=_sample_timeline(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    paths = save_pregame_identity_matchups(
        matchups=matchups,
        mirror_audit=mirror_audit,
        season=2024,
        data_root=tmp_path,
        expected_source_count=2,
    )

    assert paths["matchups_parquet"].exists()
    assert paths["metadata_json"].exists()

    metadata = json.loads(
        paths["metadata_json"].read_text(
            encoding="utf-8"
        )
    )

    assert metadata["phase"] == "2E.3A"
    assert metadata["raw_identity_edges"] == 2
    assert metadata["mirror_failures"] == 0


def test_matchup_metadata_counts():
    matchups, mirror_audit = build_pregame_identity_matchups(
        timeline=_sample_timeline(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    metadata = build_pregame_identity_matchups_metadata(
        matchups=matchups,
        mirror_audit=mirror_audit,
        season=2024,
        expected_source_count=2,
    )

    assert metadata["matchup_rows"] == 4
    assert metadata["unique_games"] == 2
    assert metadata["mirror_failures"] == 0


def test_2024_matchup_reproduction_against_reference_artifact():
    required_paths = [
        TIMELINE_PATH,
        REGISTRY_PATH,
        REFERENCE_MATCHUP_PATH,
    ]

    missing = [
        path for path in required_paths if not path.exists()
    ]

    if missing:
        pytest.skip(
            "2024 identity matchup artifacts not available in this environment."
        )

    timeline = pd.read_parquet(
        TIMELINE_PATH
    )
    registry = pd.read_csv(
        REGISTRY_PATH
    )

    built, _ = build_pregame_identity_matchups(
        timeline=timeline,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )

    reference = pd.read_parquet(
        REFERENCE_MATCHUP_PATH
    )

    assert_reproduces_reference_matchups(
        matchups=built,
        reference_matchups=reference,
    )


def test_phase_2e_identity_matchup_paths_override(tmp_path):
    paths = phase_2e_identity_matchup_paths(
        season=2024,
        data_root=tmp_path,
    )

    assert str(paths["base_dir"]).startswith(
        str(tmp_path)
    )
    assert paths["matchups_parquet"].name == "pregame_identity_matchups.parquet"
