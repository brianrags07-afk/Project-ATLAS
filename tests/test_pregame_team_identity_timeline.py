import json
from pathlib import Path

import pandas as pd
import pytest

from atlas.game_intelligence.pregame_team_identity_timeline import (
    ENGINE_VERSION,
    assert_reproduces_reference_timeline,
    build_pregame_team_identity_timeline,
    build_pregame_team_identity_timeline_metadata,
    phase_2e_team_identity_timeline_paths,
    save_pregame_team_identity_timeline,
)


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

PHASE_2D_IDENTITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "game_flow_facts"
    / "2024"
    / "team_game_flow_facts.parquet"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "pregame_identity_registry"
    / "2024"
    / "pregame_identity_source_registry.csv"
)

REFERENCE_TIMELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "game_intelligence"
    / "pregame_team_identities"
    / "2024"
    / "pregame_team_identity_timeline.parquet"
)


def _sample_source() -> pd.DataFrame:
    return pd.DataFrame({
        "game_pk": [
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        "game_date": [
            "2024-04-01",
            "2024-04-02",
            "2024-04-03",
            "2024-04-03",
            "2024-04-02",
            "2024-04-03",
        ],
        "team": [
            "NYY",
            "NYY",
            "NYY",
            "NYY",
            "BOS",
            "BOS",
        ],
        "opponent": [
            "BOS",
            "BOS",
            "BOS",
            "BOS",
            "NYY",
            "NYY",
        ],
        "home_away": [
            "HOME",
            "AWAY",
            "HOME",
            "AWAY",
            "HOME",
            "AWAY",
        ],
        "atlas_season": [2024] * 6,
        "rolling_runs": [
            1.0,
            2.0,
            3.0,
            4.0,
            8.0,
            9.0,
        ],
        "prior_whip": [
            1.10,
            1.20,
            1.30,
            1.40,
            0.95,
            1.00,
        ],
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


def test_timeline_uses_only_prior_dates_for_doubleheaders():
    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=_sample_source(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    assert failures.empty
    assert audit["audit_pass"].all()

    nyy_apr03 = timeline.loc[
        timeline["team"].eq("NYY")
        & timeline["game_date"].eq(pd.Timestamp("2024-04-03")),
        ["rolling_runs", "prior_whip"],
    ]

    assert len(nyy_apr03) == 2
    assert nyy_apr03["rolling_runs"].eq(2.0).all()
    assert nyy_apr03["prior_whip"].eq(1.20).all()


def test_timeline_paths_and_save(tmp_path):
    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=_sample_source(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    paths = save_pregame_team_identity_timeline(
        timeline=timeline,
        audit=audit,
        failures=failures,
        season=2024,
        data_root=tmp_path,
        expected_source_count=2,
    )

    assert paths["timeline_parquet"].exists()
    assert paths["audit_parquet"].exists()
    assert paths["failure_parquet"].exists()
    assert paths["metadata_json"].exists()

    metadata = json.loads(
        paths["metadata_json"].read_text(
            encoding="utf-8"
        )
    )

    assert metadata["phase"] == "2E.2"
    assert metadata["identity_features"] == 2
    assert metadata["audit_failures"] == 0


def test_timeline_metadata_counts():
    timeline, audit, failures = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=_sample_source(),
        source_registry=_sample_registry(),
        season=2024,
        expected_source_count=2,
    )

    metadata = build_pregame_team_identity_timeline_metadata(
        timeline=timeline,
        audit=audit,
        failures=failures,
        season=2024,
        expected_source_count=2,
    )

    assert metadata["team_game_rows"] == 6
    assert metadata["date_team_audit_rows"] == 5
    assert metadata["audit_failures"] == 0


def test_2024_timeline_reproduction_against_reference_artifact():
    required_paths = [
        PHASE_2D_IDENTITY_PATH,
        REGISTRY_PATH,
        REFERENCE_TIMELINE_PATH,
    ]

    missing = [
        path for path in required_paths if not path.exists()
    ]

    if missing:
        pytest.skip(
            "2024 identity timeline artifacts not available in this environment."
        )

    source = pd.read_parquet(
        PHASE_2D_IDENTITY_PATH
    )
    registry = pd.read_csv(
        REGISTRY_PATH
    )

    built, _, _ = build_pregame_team_identity_timeline(
        phase_2d_identity_frame=source,
        source_registry=registry,
        season=2024,
        expected_source_count=87,
    )

    reference = pd.read_parquet(
        REFERENCE_TIMELINE_PATH
    )

    assert_reproduces_reference_timeline(
        timeline=built,
        reference_timeline=reference,
    )


def test_phase_2e_team_identity_timeline_paths_override(tmp_path):
    paths = phase_2e_team_identity_timeline_paths(
        season=2024,
        data_root=tmp_path,
    )

    assert str(paths["base_dir"]).startswith(
        str(tmp_path)
    )
    assert paths["timeline_parquet"].name == "pregame_team_identity_timeline.parquet"
