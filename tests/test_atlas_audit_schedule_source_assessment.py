"""
Tests for atlas/audit/schedule_source_assessment.py: the required test
cases from the ATLAS no-leakage rules for schedules and series context.
"""

from __future__ import annotations

from atlas.audit.schedule_source_assessment import (
    assess_schedule_source,
    evidence_from_completed_games,
    evidence_from_published_schedule_source,
    evidence_from_series_inferred_from_results,
)


def test_completed_games_only_leaves_provenance_unknown_and_unsafe():
    evidence = [evidence_from_completed_games("master_game_database", 2024, row_count=162)]
    result = assess_schedule_source(evidence)
    assert result["provenance_status"] == "unknown"
    assert result["pregame_safety"] == "unsafe"


def test_timestamped_published_schedule_source_can_be_verified_and_safe():
    evidence = [
        evidence_from_published_schedule_source(
            "published_schedule_feed",
            2024,
            source_retrieved_at_utc="2024-03-01T00:00:00Z",
            feature_cutoff_time_utc="2024-04-01T18:00:00Z",
        )
    ]
    result = assess_schedule_source(evidence)
    assert result["provenance_status"] == "verified"
    assert result["pregame_safety"] == "safe"


def test_series_boundaries_inferred_from_results_are_unsafe():
    evidence = [evidence_from_series_inferred_from_results("master_game_database", 2024)]
    result = assess_schedule_source(evidence)
    assert result["provenance_status"] == "missing"
    assert result["pregame_safety"] == "unsafe"
    assert result["temporal_availability"] == "postgame_only"


def test_no_evidence_at_all_is_unknown_not_unsafe_or_safe():
    result = assess_schedule_source([])
    assert result["provenance_status"] == "unknown"
    assert result["pregame_safety"] == "unknown"


def test_published_schedule_source_without_cutoff_pairing_is_weaker_but_still_verified():
    evidence = [
        evidence_from_published_schedule_source(
            "published_schedule_feed", 2024, source_retrieved_at_utc=None, feature_cutoff_time_utc=None
        )
    ]
    result = assess_schedule_source(evidence)
    assert result["provenance_status"] == "verified"
