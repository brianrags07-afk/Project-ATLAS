"""
Tests for atlas/audit/report_schema.py: JSON Schema validation of the
redesigned historical readiness report.
"""

from __future__ import annotations

import json
from pathlib import Path

from atlas.audit.coverage_matrix import build_coverage_matrix
from atlas.audit.provenance import build_all_dataset_provenance
from atlas.audit.readiness import build_readiness_decisions
from atlas.audit.report_schema import EVIDENCE_SCHEMA_PATH, REPORT_SCHEMA_PATH, validate_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _profiles():
    return {
        "master_game_database": {
            "cloud_path": "data/master/master_game_database.parquet",
            "rows_by_season": {"2024": 100},
            "feature_presence": {"final_outcomes": "home_score"},
            "column_classification": {"home_score": "postgame_fact"},
            "null_percentages": {"home_score": 0.0},
            "schema_fingerprint": "fp-1",
            "data_layer": "normalized_master",
        },
    }


def test_schema_files_exist_and_parse():
    assert REPORT_SCHEMA_PATH.exists()
    assert EVIDENCE_SCHEMA_PATH.exists()
    json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
    json.loads(EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_generated_report_validates_against_schema():
    repo_inventory = {"focus_area_index": {}}
    profiles = _profiles()
    matrix = build_coverage_matrix(profiles, repo_inventory)
    provenance = build_all_dataset_provenance(profiles, cloud_inventory=None)
    readiness = build_readiness_decisions(matrix, profiles, provenance)
    errors = validate_report(matrix, readiness)
    assert errors == [], errors


def test_invalid_report_is_rejected():
    bad_readiness = {"generated_at_utc": "not-a-real-timestamp", "decisions": {}, "note": "x"}
    errors = validate_report([], bad_readiness)
    assert errors, "an invalid date-time should be rejected by the schema"
