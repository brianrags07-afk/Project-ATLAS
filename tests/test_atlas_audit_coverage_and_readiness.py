"""
Tests for atlas/audit/coverage_matrix.py and atlas/audit/readiness.py.
"""

from __future__ import annotations

from atlas.audit.coverage_matrix import (
    ALLOWED_STATUSES,
    build_coverage_matrix,
)
from atlas.audit.readiness import ALLOWED_DECISIONS, build_readiness_decisions

REPO_INVENTORY_FIXTURE = {
    "focus_area_index": {
        "memories": ["atlas/memory/team_memory.py"],
        "identities": ["atlas/identities/identity_engine.py"],
        "concepts": [],
        "validation": ["atlas/validation/validation_engine.py"],
        "prediction": ["atlas/predictions/prediction_engine.py"],
        "pregame_snapshots": ["atlas/pregame/canonical_core_evidence_matrix.py"],
    }
}


def _dataset_profiles_fixture():
    return {
        "master_game_database": {
            "rows_by_season": {"2024": 100, "2025": 50},
            "feature_presence": {
                "game_pk": "game_pk",
                "scheduled_first_pitch": None,
                "final_outcomes": "home_score",
                "starter_information": None,
                "bullpen_usage": None,
                "lineups": None,
                "injuries": None,
                "weather": None,
                "venue": "venue",
                "umpire": None,
                "rest": None,
                "travel": None,
                "published_series_context": None,
                "market_data": None,
            },
            "column_classification": {
                "game_pk": "identifier",
                "home_score": "postgame_fact",
                "venue": "schedule_safe",
            },
        },
        "master_pitch_database": {
            "rows_by_season": {"2024": 1000},
            "pitches_by_season": {"2024": 1000},
        },
    }


def test_build_coverage_matrix_uses_only_allowed_statuses():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    for row in matrix:
        assert row["status"] in ALLOWED_STATUSES
        assert row["evidence"]


def test_build_coverage_matrix_published_schedule_complete_for_seasons_with_rows():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row_2024 = next(r for r in matrix if r["row"] == "published_schedule" and r["season"] == 2024)
    assert row_2024["status"] == "complete"
    row_2026 = next(r for r in matrix if r["row"] == "published_schedule" and r["season"] == 2026)
    assert row_2026["status"] == "missing"


def test_build_coverage_matrix_missing_when_no_column_evidence():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = next(r for r in matrix if r["row"] == "starters" and r["season"] == 2024)
    assert row["status"] == "missing"


def test_build_coverage_matrix_pitch_by_pitch_marked_not_pregame_safe():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = next(r for r in matrix if r["row"] == "pitch_by_pitch" and r["season"] == 2024)
    assert row["status"] == "present_but_not_pregame_safe"


def test_build_coverage_matrix_concept_discovery_missing_when_no_modules():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = next(r for r in matrix if r["row"] == "concept_discovery" and r["season"] == 2024)
    assert row["status"] == "missing"


def test_build_readiness_decisions_covers_all_seven_questions():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    decisions = readiness["decisions"]
    assert len(decisions) == 7
    for key, entry in decisions.items():
        assert entry["decision"] in ALLOWED_DECISIONS
        assert "evidence" in entry
        assert "missing_requirements" in entry
        assert "risks" in entry
        assert "next_action" in entry


def test_readiness_2025_backtest_not_ready_without_pregame_proof():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    backtest = readiness["decisions"]["E_2025_walk_forward_backtest"]
    assert backtest["decision"] in ("not_ready", "unknown")
    assert any("leak" in risk.lower() for risk in backtest["risks"])


def test_readiness_note_states_full_table_is_not_automatically_pregame_safe():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    assert "not automatically pregame-safe" in readiness["note"]
