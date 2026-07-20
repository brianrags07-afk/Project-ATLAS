"""
Validity tests for schemas/pipeline_manifest.schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "pipeline_manifest.schema.json"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _valid_manifest() -> dict:
    return {
        "pipeline_run_id": "run_2025_000001",
        "run_mode": "backtest_2025",
        "season": 2025,
        "commit_sha": "abcdef1",
        "pipeline_version": "1.0.0",
        "source_objects": [
            {
                "path": "gs://atlas-mlb-data-brian-4817/data/master/master_game_database.parquet",
                "hash": "a" * 64,
                "hash_algorithm": "sha256",
                "date_range": {"min_date": "2025-03-27", "max_date": "2025-04-01"},
                "season_range": [2025],
                "row_count": 100,
                "column_count": 12,
                "schema_ref": "schemas/master_game_database.schema.json",
            }
        ],
        "output_objects": [
            {
                "path": "staging/pregame_game_cards_2025.parquet",
                "hash": "b" * 64,
                "hash_algorithm": "sha256",
                "row_count": 100,
                "column_count": 40,
                "schema_ref": "schemas/pregame_game_card.schema.json",
                "promotion_status": "staging",
            }
        ],
        "started_at_utc": "2025-04-01T12:00:00Z",
        "completed_at_utc": "2025-04-01T12:30:00Z",
        "run_extent": "incremental",
        "validation_results": {"status": "passed"},
        "leakage_audit_results": {"status": "passed"},
        "promotion_status": "staging",
        "error_status": "ok",
    }


def test_schema_file_exists_and_is_valid_json():
    assert SCHEMA_PATH.exists()
    schema = _load_schema()
    assert schema["$schema"].startswith("https://json-schema.org/")


def test_schema_requires_core_manifest_fields():
    schema = _load_schema()
    required = set(schema["required"])
    for key in (
        "pipeline_run_id",
        "run_mode",
        "season",
        "commit_sha",
        "pipeline_version",
        "source_objects",
        "output_objects",
        "started_at_utc",
        "validation_results",
        "leakage_audit_results",
        "promotion_status",
        "error_status",
    ):
        assert key in required


def test_schema_run_mode_enum_covers_expected_modes():
    schema = _load_schema()
    assert set(schema["properties"]["run_mode"]["enum"]) == {
        "discovery_2024",
        "backtest_2025",
        "forward_2026",
        "rebuild",
        "audit",
    }


def test_schema_output_objects_default_to_staging_enum_option():
    schema = _load_schema()
    output_item_schema = schema["properties"]["output_objects"]["items"]
    assert "staging" in output_item_schema["properties"]["promotion_status"]["enum"]


def test_valid_manifest_fixture_has_all_required_keys():
    schema = _load_schema()
    manifest = _valid_manifest()
    for key in schema["required"]:
        assert key in manifest, f"missing required key: {key}"


def test_valid_manifest_fixture_source_objects_have_all_required_subkeys():
    schema = _load_schema()
    required_subkeys = set(schema["properties"]["source_objects"]["items"]["required"])
    manifest = _valid_manifest()
    for entry in manifest["source_objects"]:
        assert required_subkeys.issubset(entry)
