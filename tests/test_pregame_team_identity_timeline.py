"""
Fixture and production tests for the Phase 2E.2 strict prior-date pregame
team identity timeline builder.

Fixture/unit tests run entirely against the compact ATLAS contract pack
shipped in the repository at ``atlas_reference/``. The production-only
integration test is skipped (not failed) when the full production Google
Drive workspace is unavailable.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas.config import DATA_ROOT
from atlas.game_intelligence.pregame_identity_source_registry import (
    build_pregame_identity_source_registry,
)
from atlas.game_intelligence.pregame_team_identity_timeline import (
    ENGINE_VERSION,
    assert_reproduces_reference_timeline,
    build_pregame_team_identity_timeline,
    build_pregame_team_identity_timeline_metadata,
    phase_2e_team_identity_timeline_paths,
    save_pregame_team_identity_timeline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_GAMES_FIXTURE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "games"
    / "data__game_intelligence__game_flow_facts__2024__team_game_flow_facts.parquet.games.parquet"
)

CONTRACT_TIMELINE_FIXTURE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "general"
    / "data__game_intelligence__pregame_team_identities__2024__pregame_team_identity_timeline.parquet.sample.parquet"
)

CONTRACT_TIMELINE_AUDIT_FIXTURE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "general"
    / "data__game_intelligence__pregame_team_identities__2024__pregame_team_identity_timeline_audit.parquet.sample.parquet"
)

PHASE_2D_IDENTITY_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "game_flow_facts"
    / "2024"
    / "team_game_flow_facts.parquet"
)

REGISTRY_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "pregame_identity_registry"
    / "2024"
    / "pregame_identity_source_registry.csv"
)

REFERENCE_TIMELINE_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "pregame_team_identities"
    / "2024"
    / "pregame_team_identity_timeline.parquet"
)


def _contract_pack_games_frame() -> pd.DataFrame:
    return pd.read_parquet(CONTRACT_GAMES_FIXTURE)


def _contract_pack_registry() -> pd.DataFrame:
    return build_pregame_identity_source_registry(
        _contract_pack_games_frame(),
        season=2024,
    )


def test_engine_version():
    assert ENGINE_VERSION == "1.0.0"


def test_timeline_uses_only_prior_dates_for_doubleheaders():
    """Synthetic doubleheader check: two same-team games on one calendar
    date must not see each other, and the next date's expanding mean must
    equal the strict average of exactly the two prior games."""
    source = pd.DataFrame({
        "game_pk": [1, 2, 3, 4],
        "game_date": [
            "2024-04-01",
            "2024-04-02",
            "2024-04-03",
            "2024-04-03",
        ],
        "team": ["NYY", "NYY", "NYY", "NYY"],
        "opponent": ["BOS", "BOS", "BOS", "BOS"],
        "home_away": ["HOME", "AWAY", "HOME", "AWAY"],
        "atlas_season": [2024] * 4,
        "team_score": [1.0, 2.0, 3.0, 4.0],
        "won": [1, 1, 0, 0],
    })

    registry = pd.DataFrame({
        "column": ["team_score", "won"],
        "source_status": ["lagged_identity_source"] * 2,
    })

    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=2,
    )

    assert failures.empty
    assert audit["audit_pass"].all()

    apr03_rows = timeline.loc[
        timeline["game_date"].eq(pd.Timestamp("2024-04-03"))
    ]
    assert len(apr03_rows) == 2
    assert apr03_rows["identity_games_before_date"].eq(2).all()
    assert apr03_rows["identity__expanding_mean__team_score"].eq(1.5).all()
    assert apr03_rows["identity_sample_1_plus"].all()
    assert not apr03_rows["identity_sample_5_plus"].any()


def test_timeline_first_game_has_no_history():
    source = pd.DataFrame({
        "game_pk": [1],
        "game_date": ["2024-04-01"],
        "team": ["NYY"],
        "opponent": ["BOS"],
        "home_away": ["HOME"],
        "atlas_season": [2024],
        "team_score": [1.0],
    })
    registry = pd.DataFrame({
        "column": ["team_score"],
        "source_status": ["lagged_identity_source"],
    })

    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=1,
    )

    assert timeline["identity_games_before_date"].eq(0).all()
    assert pd.isna(
        timeline["identity__expanding_mean__team_score"]
    ).all()
    assert not timeline["identity_sample_1_plus"].any()


def test_build_timeline_matches_authoritative_contract_pack_fixture():
    """Cross-check every overlapping (game_pk, team) row of the built
    timeline against the authoritative contract-pack fixture sample."""
    source = _contract_pack_games_frame()
    registry = _contract_pack_registry()

    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )

    reference = pd.read_parquet(CONTRACT_TIMELINE_FIXTURE)

    assert list(timeline.columns) == list(reference.columns)

    merged = timeline.merge(
        reference,
        on=["game_pk", "team"],
        suffixes=("_built", "_ref"),
    )
    assert len(merged) == len(reference)

    assert (
        merged["identity_games_before_date_built"]
        == merged["identity_games_before_date_ref"]
    ).all()
    assert (
        merged["identity_dates_before_date_built"]
        == merged["identity_dates_before_date_ref"]
    ).all()

    sample_feature_columns = [
        "identity__expanding_mean__team_score",
        "identity__expanding_mean__won",
        "identity__expanding_mean__run_differential",
    ]
    for column in sample_feature_columns:
        assert np.allclose(
            merged[f"{column}_built"].fillna(-999.0),
            merged[f"{column}_ref"].fillna(-999.0),
            atol=1e-6,
        ), f"Mismatch in {column} against authoritative fixture."


def test_build_timeline_matches_authoritative_audit_fixture():
    source = _contract_pack_games_frame()
    registry = _contract_pack_registry()

    _timeline, audit, _failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )

    reference_audit = pd.read_parquet(CONTRACT_TIMELINE_AUDIT_FIXTURE)

    assert list(audit.columns) == list(reference_audit.columns)

    merged = audit.merge(
        reference_audit,
        on=["team", "game_date"],
        suffixes=("_built", "_ref"),
    )
    assert len(merged) > 0

    for column in [
        "target_team_game_rows",
        "expected_prior_games",
        "observed_prior_games",
        "representative_feature_checks",
        "representative_feature_passes",
        "audit_pass",
    ]:
        assert (
            merged[f"{column}_built"] == merged[f"{column}_ref"]
        ).all(), f"Mismatch in audit column '{column}'."


def test_timeline_paths_and_save(tmp_path):
    source = pd.DataFrame({
        "game_pk": [1, 2],
        "game_date": ["2024-04-01", "2024-04-02"],
        "team": ["NYY", "NYY"],
        "opponent": ["BOS", "BOS"],
        "home_away": ["HOME", "AWAY"],
        "atlas_season": [2024, 2024],
        "team_score": [1.0, 2.0],
    })
    registry = pd.DataFrame({
        "column": ["team_score"],
        "source_status": ["lagged_identity_source"],
    })

    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=1,
    )

    paths = save_pregame_team_identity_timeline(
        timeline=timeline,
        audit=audit,
        failures=failures,
        season=2024,
        data_root=tmp_path,
        expected_source_count=1,
    )

    assert paths["timeline_parquet"].exists()
    assert paths["audit_parquet"].exists()
    assert paths["failure_parquet"].exists()
    assert paths["metadata_json"].exists()

    metadata = json.loads(
        paths["metadata_json"].read_text(encoding="utf-8")
    )

    assert metadata["phase"] == "2E.2"
    assert metadata["identity_features"] == 1
    assert metadata["audit_failures"] == 0


def test_2024_timeline_reproduction_against_production_workspace():
    """Production-only integration test.

    Skipped (not failed) when the full production ATLAS Google Drive
    workspace is unavailable in this environment.
    """
    required_paths = [
        PHASE_2D_IDENTITY_PATH,
        REGISTRY_PATH,
        REFERENCE_TIMELINE_PATH,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        pytest.skip(
            "2024 identity timeline artifacts not available in this environment."
        )

    source = pd.read_parquet(PHASE_2D_IDENTITY_PATH)
    registry = pd.read_csv(REGISTRY_PATH)

    built, _, _ = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )

    reference = pd.read_parquet(REFERENCE_TIMELINE_PATH)

    assert_reproduces_reference_timeline(
        timeline=built,
        reference_timeline=reference,
    )


def test_phase_2e_team_identity_timeline_paths_override(tmp_path):
    paths = phase_2e_team_identity_timeline_paths(
        season=2024,
        data_root=tmp_path,
    )

    assert str(paths["base_dir"]).startswith(str(tmp_path))
    assert (
        paths["timeline_parquet"].name
        == "pregame_team_identity_timeline.parquet"
    )
