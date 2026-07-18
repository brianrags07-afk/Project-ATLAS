"""
Fixture and production tests for the Phase 2E.1 pregame identity source
registry builder.

Fixture/unit tests below run entirely against the compact ATLAS contract
pack shipped in the repository at ``atlas_reference/`` and require no
production Google Drive workspace. The single production-only test at the
bottom of this file is explicitly skipped (not failed) when the production
workspace is unavailable in this environment.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from atlas.config import DATA_ROOT
from atlas.game_intelligence.pregame_identity_source_registry import (
    ENGINE_VERSION,
    EXPECTED_LAGGED_SOURCE_COUNT,
    EXPECTED_TOTAL_COLUMNS,
    IDENTITY_SOURCE_CLASSIFICATION,
    approved_lagged_identity_columns,
    assert_matches_frozen_contract,
    build_pregame_identity_source_registry,
    normalize_phase_2d_identity_inputs,
    phase_2e_identity_source_registry_paths,
    save_pregame_identity_source_registry,
    validate_pregame_identity_source_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_GAMES_FIXTURE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "games"
    / "data__game_intelligence__game_flow_facts__2024__team_game_flow_facts.parquet.games.parquet"
)

CONTRACT_REGISTRY_FIXTURE = (
    REPO_ROOT
    / "atlas_reference"
    / "samples"
    / "general"
    / "data__game_intelligence__pregame_identity_registry__2024__pregame_identity_source_registry.csv.sample.parquet"
)

REGISTRY_PATH = (
    DATA_ROOT
    / "game_intelligence"
    / "pregame_identity_registry"
    / "2024"
    / "pregame_identity_source_registry.csv"
)


def _contract_pack_games_frame() -> pd.DataFrame:
    return pd.read_parquet(CONTRACT_GAMES_FIXTURE)


def _contract_pack_registry_fixture() -> pd.DataFrame:
    return pd.read_parquet(CONTRACT_REGISTRY_FIXTURE)


def test_engine_version():
    assert ENGINE_VERSION == "1.0.0"


def test_classification_table_has_exact_frozen_column_count():
    assert len(IDENTITY_SOURCE_CLASSIFICATION) == EXPECTED_TOTAL_COLUMNS

    approved = [
        column
        for column, (
            _family,
            source_status,
            _same_game_safe,
            _requires_shift,
            _historical_aggregation_allowed,
        ) in IDENTITY_SOURCE_CLASSIFICATION.items()
        if source_status == "lagged_identity_source"
    ]
    assert len(approved) == EXPECTED_LAGGED_SOURCE_COUNT


def test_assert_matches_frozen_contract_rejects_unknown_columns():
    frame = pd.DataFrame({
        "game_pk": [1],
        "team": ["NYY"],
        "invented_stat": [1.0],
    })

    with pytest.raises(KeyError):
        assert_matches_frozen_contract(frame)


def test_assert_matches_frozen_contract_rejects_missing_columns():
    frame = pd.DataFrame({
        "game_pk": [1],
        "team": ["NYY"],
    })

    with pytest.raises(KeyError):
        assert_matches_frozen_contract(frame)


def test_normalize_phase_2d_identity_inputs_uppercases_team_codes():
    frame = _contract_pack_games_frame().head(4).copy()
    normalized = normalize_phase_2d_identity_inputs(
        frame,
        season=2024,
    )

    assert normalized["team"].str.isupper().all()
    assert normalized["game_pk"].dtype == "int64"


def test_normalize_phase_2d_identity_inputs_rejects_duplicate_team_games():
    frame = _contract_pack_games_frame().head(2).copy()
    frame = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(AssertionError):
        normalize_phase_2d_identity_inputs(
            frame,
            season=2024,
        )


def test_build_registry_matches_authoritative_contract_pack_fixture():
    """The registry built from the real Phase 2D fixture must be
    byte-identical (column names, order, and every governance field) to the
    authoritative registry fixture shipped in the contract pack."""
    source = _contract_pack_games_frame()

    registry = build_pregame_identity_source_registry(
        source,
        season=2024,
    )

    reference = _contract_pack_registry_fixture()

    assert list(registry.columns) == list(reference.columns)
    assert len(registry) == len(reference) == EXPECTED_TOTAL_COLUMNS

    for column in [
        "column",
        "family",
        "source_status",
        "same_game_safe",
        "requires_shift",
        "historical_aggregation_allowed",
        "reason",
    ]:
        assert registry[column].tolist() == reference[column].tolist(), (
            f"Mismatch in column '{column}' against authoritative fixture."
        )

    approved = approved_lagged_identity_columns(registry)
    assert len(approved) == EXPECTED_LAGGED_SOURCE_COUNT


def test_build_registry_rejects_wrong_total_column_count():
    source = _contract_pack_games_frame().drop(
        columns=["role_deficit_reductions"]
    )

    with pytest.raises(KeyError):
        build_pregame_identity_source_registry(
            source,
            season=2024,
        )


def test_validate_registry_rejects_same_game_safe_lagged_source():
    source = _contract_pack_games_frame()
    registry = build_pregame_identity_source_registry(
        source,
        season=2024,
    )

    lagged_index = registry.index[
        registry["source_status"].eq("lagged_identity_source")
    ][0]
    registry.loc[lagged_index, "same_game_safe"] = True

    with pytest.raises(AssertionError):
        validate_pregame_identity_source_registry(
            registry,
            expected_source_count=EXPECTED_LAGGED_SOURCE_COUNT,
        )


def test_save_registry_writes_csv_and_metadata(tmp_path):
    source = _contract_pack_games_frame()
    registry = build_pregame_identity_source_registry(
        source,
        season=2024,
    )

    paths = save_pregame_identity_source_registry(
        registry,
        season=2024,
        data_root=tmp_path,
    )

    assert paths["registry_csv"].exists()
    assert paths["metadata_json"].exists()

    metadata = json.loads(
        paths["metadata_json"].read_text(
            encoding="utf-8"
        )
    )
    assert metadata["total_columns"] == EXPECTED_TOTAL_COLUMNS
    assert metadata["approved_lagged_identity_sources"] == EXPECTED_LAGGED_SOURCE_COUNT
    assert metadata["same_game_identity_sources"] == 0


def test_registry_paths_use_centralized_data_root_when_overridden(tmp_path):
    paths = phase_2e_identity_source_registry_paths(
        season=2024,
        data_root=tmp_path,
    )

    assert str(paths["base_dir"]).startswith(
        str(tmp_path)
    )
    assert paths["registry_csv"].name == "pregame_identity_source_registry.csv"


def test_2024_registry_reproduction_against_production_workspace():
    """Production-only integration test.

    Skipped (not failed) when the full ``/content/drive/MyDrive/
    Project_Atlas`` production workspace is unavailable, per ATLAS
    governance: production-only checks must never be counted as code
    failures in the GitHub agent sandbox.
    """
    if not REGISTRY_PATH.exists():
        pytest.skip(
            "Production ATLAS workspace is unavailable in this environment; "
            f"expected {REGISTRY_PATH}."
        )

    production_registry = pd.read_csv(REGISTRY_PATH)

    assert list(production_registry.columns) == [
        "column",
        "dtype",
        "family",
        "source_status",
        "same_game_safe",
        "requires_shift",
        "historical_aggregation_allowed",
        "non_null_rows",
        "unique_values",
        "reason",
    ]
    assert len(production_registry) == EXPECTED_TOTAL_COLUMNS
