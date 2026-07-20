"""
Tests for atlas/audit/job_summary.py against the redesigned coverage
matrix / readiness report shapes.
"""

from __future__ import annotations

from atlas.audit.coverage_matrix import build_coverage_matrix
from atlas.audit.job_summary import render_job_summary
from atlas.audit.provenance import build_all_dataset_provenance
from atlas.audit.readiness import build_readiness_decisions

CLOUD_INVENTORY_FIXTURE = {
    "bucket": "gs://atlas-mlb-data-brian-4817",
    "object_count": 4,
    "known_master_files_found": ["data/master/master_game_database.parquet"],
    "known_master_files_missing": ["data/master/team_game_state.parquet"],
}


def _profiles():
    return {
        "master_game_database": {
            "cloud_path": "data/master/master_game_database.parquet",
            "rows_by_season": {"2024": 100},
            "seasons_present": ["2024"],
            "feature_presence": {"final_outcomes": "home_score"},
            "column_classification": {"home_score": "postgame_fact"},
            "null_percentages": {"home_score": 0.0},
            "schema_fingerprint": "fp-1",
            "data_layer": "normalized_master",
            "data_layer_note": "not proof of raw source completeness",
            "data_layer_confidence": "heuristic",
        },
    }


def test_render_job_summary_includes_data_layers_and_leakage_and_next_step():
    repo_inventory = {"focus_area_index": {}}
    profiles = _profiles()
    matrix = build_coverage_matrix(profiles, repo_inventory)
    provenance = build_all_dataset_provenance(profiles, cloud_inventory=None)
    readiness = build_readiness_decisions(matrix, profiles, provenance)
    summary = render_job_summary(CLOUD_INVENTORY_FIXTURE, profiles, matrix, readiness)

    assert "Data layers observed" in summary
    assert "normalized_master" in summary
    assert "Major leakage risks" in summary
    assert "Exact recommended next step" in summary
    assert "authorizes any of those actions" in summary


def test_render_job_summary_never_claims_read_write_mutation():
    repo_inventory = {"focus_area_index": {}}
    profiles = _profiles()
    matrix = build_coverage_matrix(profiles, repo_inventory)
    readiness = build_readiness_decisions(matrix, profiles)
    summary = render_job_summary(CLOUD_INVENTORY_FIXTURE, profiles, matrix, readiness)
    assert "No Cloud Storage object was deleted, overwritten, renamed" in summary
