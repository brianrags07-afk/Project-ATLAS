"""
Fixture and production tests for the Phase 2E.3A pregame team-versus-
opponent identity matchup builder.

Fixture/unit tests run entirely against the compact ATLAS contract pack
shipped in the repository at ``atlas_reference/``. No fixture sample exists
for the matchup artifact itself (schema-profile-only ground truth), so
fixture tests here validate exact column ordering against the authoritative
schema plus internal consistency invariants (mirroring, sample-size
diagnostics). The production-only integration test is skipped (not failed)
when the full production Google Drive workspace is unavailable.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from atlas.config import DATA_ROOT
from atlas.game_intelligence.pregame_identity_source_registry import (
    build_pregame_identity_source_registry,
)
from atlas.game_intelligence.pregame_team_identity_timeline import (
    build_pregame_team_identity_timeline,
)
from atlas.game_intelligence.pregame_identity_matchup_builder import (
    ENGINE_VERSION,
    assert_reproduces_reference_matchups,
    build_pregame_identity_matchups,
    build_pregame_identity_matchups_metadata,
    phase_2e_identity_matchup_paths,
    save_pregame_identity_matchups,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_GAMES_FIXTURE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "games"
    / "data__game_intelligence__game_flow_facts__2024__team_game_flow_facts.parquet.games.parquet"
)

CONTRACT_MATCHUP_SCHEMA = (
    REPO_ROOT
    / "atlas_reference"
    / "schemas"
    / "data__game_intelligence__pregame_identity_matchups__2024__pregame_identity_matchups.parquet.schema.json"
)

TIMELINE_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "pregame_team_identities"
    / "2024"
    / "pregame_team_identity_timeline.parquet"
)

REGISTRY_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "pregame_identity_registry"
    / "2024"
    / "pregame_identity_source_registry.csv"
)

REFERENCE_MATCHUP_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "pregame_identity_matchups"
    / "2024"
    / "pregame_identity_matchups.parquet"
)


def _contract_pack_matchups() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(CONTRACT_GAMES_FIXTURE)
    registry = build_pregame_identity_source_registry(
        source,
        season=2024,
    )
    timeline, _audit, _failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )
    matchups, mirror_audit = build_pregame_identity_matchups(
        timeline=timeline,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )
    return matchups, mirror_audit


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
        "identity_games_before_date": [10, 12, 11, 13],
        "identity_dates_before_date": [10, 12, 11, 13],
        "identity__expanding_mean__rolling_runs": [5.0, 3.0, 4.5, 3.5],
        "identity__expanding_mean__prior_whip": [1.05, 1.20, 1.10, 1.15],
    })


def _sample_registry() -> pd.DataFrame:
    return pd.DataFrame({
        "column": ["rolling_runs", "prior_whip"],
        "source_status": ["lagged_identity_source"] * 2,
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
    assert matchups["all_identity_edges_mirror"].all()


def test_matchup_sample_diagnostics():
    matchups, _mirror_audit = build_pregame_identity_matchups(
        timeline=_sample_timeline(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    nyy_game_100 = matchups.loc[
        matchups["game_pk"].eq(100) & matchups["team"].eq("NYY")
    ].iloc[0]

    assert nyy_game_100["minimum_identity_games"] == 10
    assert nyy_game_100["maximum_identity_games"] == 12
    assert nyy_game_100["identity_game_sample_gap"] == -2
    assert nyy_game_100["identity_sample_confidence_label"] == "MODERATE"
    assert nyy_game_100["available_identity_edges"] == 2
    assert nyy_game_100["missing_identity_edges"] == 0


def test_matchup_confidence_label_no_history():
    timeline = _sample_timeline()
    timeline["identity_games_before_date"] = [0, 0, 1, 1]

    matchups, _mirror_audit = build_pregame_identity_matchups(
        timeline=timeline,
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    no_history = matchups.loc[matchups["game_pk"].eq(100)]
    assert no_history["identity_sample_confidence_label"].eq("NO_HISTORY").all()
    assert no_history["identity_sample_balance"].eq(1.0).all()


def test_matchup_column_order_matches_authoritative_schema():
    """No fixture row sample exists for matchups; verify exact column
    ordering against the authoritative schema profile instead."""
    matchups, _mirror_audit = _contract_pack_matchups()

    schema = json.loads(
        CONTRACT_MATCHUP_SCHEMA.read_text(encoding="utf-8")
    )
    schema_columns = list(schema["columns"].keys())

    assert list(matchups.columns) == schema_columns


def test_matchup_builder_mirror_audit_passes_on_contract_pack_fixture():
    matchups, mirror_audit = _contract_pack_matchups()

    assert not matchups.empty
    assert mirror_audit["audit_pass"].all()
    assert matchups["all_identity_edges_mirror"].all()


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
        paths["metadata_json"].read_text(encoding="utf-8")
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


def test_2024_matchup_reproduction_against_production_workspace():
    """Production-only integration test.

    Skipped (not failed) when the full production ATLAS Google Drive
    workspace is unavailable in this environment.
    """
    required_paths = [
        TIMELINE_PATH,
        REGISTRY_PATH,
        REFERENCE_MATCHUP_PATH,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        pytest.skip(
            "2024 identity matchup artifacts not available in this environment."
        )

    timeline = pd.read_parquet(TIMELINE_PATH)
    registry = pd.read_csv(REGISTRY_PATH)

    built, _ = build_pregame_identity_matchups(
        timeline=timeline,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )

    reference = pd.read_parquet(REFERENCE_MATCHUP_PATH)

    assert_reproduces_reference_matchups(
        matchups=built,
        reference_matchups=reference,
    )


def test_phase_2e_identity_matchup_paths_override(tmp_path):
    paths = phase_2e_identity_matchup_paths(
        season=2024,
        data_root=tmp_path,
    )

    assert str(paths["base_dir"]).startswith(str(tmp_path))
    assert (
        paths["matchups_parquet"].name
        == "pregame_identity_matchups.parquet"
    )
