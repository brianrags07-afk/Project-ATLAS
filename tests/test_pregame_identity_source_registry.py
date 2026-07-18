import json

import pandas as pd
import pytest

from atlas.game_intelligence.pregame_identity_source_registry import (
    ENGINE_VERSION,
    build_pregame_identity_source_registry,
    normalize_phase_2d_identity_inputs,
    phase_2e_identity_source_registry_paths,
    save_pregame_identity_source_registry,
    validate_pregame_identity_source_registry,
)


def _phase_2d_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "game_pk": [1001, 1002],
        "game_date": ["2024-04-01", "2024-04-02"],
        "team": ["nyy", "bos"],
        "opponent": ["BOS", "NYY"],
        "atlas_season": [2024, 2024],
        "rolling_runs_scored": [4.2, 3.8],
        "prior_bullpen_whip": [1.22, 1.09],
        "lag_team_obp": [0.329, 0.311],
        "target_team_win": [1, 0],
    })


def test_normalize_phase_2d_identity_inputs_uppercases_team_codes():
    normalized = normalize_phase_2d_identity_inputs(
        _phase_2d_sample(),
        season=2024,
    )

    assert normalized["team"].tolist() == ["NYY", "BOS"]
    assert normalized["game_pk"].dtype == "int64"


def test_normalize_phase_2d_identity_inputs_rejects_duplicate_team_games():
    frame = pd.concat(
        [
            _phase_2d_sample(),
            _phase_2d_sample().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(AssertionError):
        normalize_phase_2d_identity_inputs(
            frame,
            season=2024,
        )


def test_build_registry_uses_explicit_approved_columns():
    registry = build_pregame_identity_source_registry(
        _phase_2d_sample(),
        season=2024,
        expected_source_count=3,
        approved_columns=[
            "rolling_runs_scored",
            "prior_bullpen_whip",
            "lag_team_obp",
        ],
    )

    assert len(registry) == 3
    assert registry["same_game_source"].eq(False).all()
    assert registry["future_games_used"].eq(False).all()
    assert registry["registry_engine_version"].eq(ENGINE_VERSION).all()


def test_build_registry_auto_selection_ignores_leakage_columns():
    registry = build_pregame_identity_source_registry(
        _phase_2d_sample(),
        season=2024,
        expected_source_count=3,
    )

    assert "target_team_win" not in set(
        registry["identity_feature_name"]
    )


def test_validate_registry_rejects_same_game_sources():
    registry = build_pregame_identity_source_registry(
        _phase_2d_sample(),
        season=2024,
        expected_source_count=3,
    )
    registry.loc[
        registry.index[0],
        "same_game_source",
    ] = True

    with pytest.raises(AssertionError):
        validate_pregame_identity_source_registry(
            registry,
            expected_source_count=3,
        )


def test_save_registry_writes_csv_and_metadata(tmp_path):
    registry = build_pregame_identity_source_registry(
        _phase_2d_sample(),
        season=2024,
        expected_source_count=3,
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
    assert metadata["source_count"] == 3
    assert metadata["same_game_sources"] == 0
    assert metadata["future_games_used_sources"] == 0


def test_registry_paths_use_centralized_data_root_when_overridden(tmp_path):
    paths = phase_2e_identity_source_registry_paths(
        season=2024,
        data_root=tmp_path,
    )

    assert str(paths["base_dir"]).startswith(
        str(tmp_path)
    )
    assert paths["registry_csv"].name == "pregame_identity_source_registry.csv"
