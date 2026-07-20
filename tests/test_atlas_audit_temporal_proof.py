"""
Tests for atlas/audit/temporal_proof.py: per-game timestamp proof and the
explicit temporal_availability -> pregame_safety mapping.
"""

from __future__ import annotations

from atlas.audit.evidence import make_evidence
from atlas.audit.temporal_proof import (
    assess_field_temporal_availability,
    assess_pregame_safety_from_temporal_availability,
    has_per_game_timestamp_proof,
)


def test_has_per_game_timestamp_proof_true_when_retrieved_before_cutoff():
    assert has_per_game_timestamp_proof("2024-04-01T10:00:00Z", "2024-04-01T18:00:00Z") is True


def test_has_per_game_timestamp_proof_false_when_retrieved_after_cutoff():
    assert has_per_game_timestamp_proof("2024-04-01T20:00:00Z", "2024-04-01T18:00:00Z") is False


def test_has_per_game_timestamp_proof_false_when_missing():
    assert has_per_game_timestamp_proof(None, "2024-04-01T18:00:00Z") is False
    assert has_per_game_timestamp_proof("2024-04-01T10:00:00Z", None) is False


def test_no_evidence_records_yields_unknown_temporal_availability():
    availability, _reason = assess_field_temporal_availability([])
    assert availability == "unknown"


def test_storage_upload_timestamp_alone_is_not_pregame_proof():
    records = [
        make_evidence(
            "storage_upload_timestamp",
            source="master_game_database",
            observed_value={"updated": "2024-01-01T00:00:00Z"},
            confidence="observed",
        )
    ]
    availability, reason = assess_field_temporal_availability(records)
    assert availability == "unknown"
    assert "NOT proof" in reason


def test_pregame_proof_with_per_game_cutoff_pairing():
    records = [
        make_evidence(
            "source_retrieved_at_timestamp",
            source="starter_feed",
            observed_value={
                "source_retrieved_at_utc": "2024-04-01T10:00:00Z",
                "feature_cutoff_time_utc": "2024-04-01T18:00:00Z",
            },
            confidence="observed",
        )
    ]
    availability, _reason = assess_field_temporal_availability(records)
    assert availability == "pregame_proven"


def test_pregame_proof_rejected_when_retrieved_after_cutoff():
    records = [
        make_evidence(
            "source_retrieved_at_timestamp",
            source="starter_feed",
            observed_value={
                "source_retrieved_at_utc": "2024-04-01T20:00:00Z",
                "feature_cutoff_time_utc": "2024-04-01T18:00:00Z",
            },
            confidence="observed",
        )
    ]
    availability, reason = assess_field_temporal_availability(records)
    assert availability == "unknown"
    assert "not on-or-before" in reason


def test_postgame_only_evidence_yields_postgame_only():
    records = [make_evidence("completed_game_record", source="master_game_database", confidence="observed")]
    availability, _reason = assess_field_temporal_availability(records)
    assert availability == "postgame_only"


def test_mixed_evidence_yields_mixed():
    records = [
        make_evidence(
            "source_retrieved_at_timestamp",
            source="starter_feed",
            observed_value={
                "source_retrieved_at_utc": "2024-04-01T10:00:00Z",
                "feature_cutoff_time_utc": "2024-04-01T18:00:00Z",
            },
            confidence="observed",
        ),
        make_evidence("completed_game_record", source="master_game_database", confidence="observed"),
    ]
    availability, _reason = assess_field_temporal_availability(records)
    assert availability == "mixed"


def test_pregame_safety_mapping_is_explicit_and_documented():
    assert assess_pregame_safety_from_temporal_availability("pregame_proven") == "safe"
    assert assess_pregame_safety_from_temporal_availability("mixed") == "conditional"
    assert assess_pregame_safety_from_temporal_availability("postgame_only") == "unsafe"
    assert assess_pregame_safety_from_temporal_availability("unknown") == "unknown"
    assert assess_pregame_safety_from_temporal_availability("not_applicable") == "not_applicable"


def test_pregame_safety_not_applicable_for_non_dynamic_field_regardless_of_temporal_state():
    assert (
        assess_pregame_safety_from_temporal_availability("pregame_proven", is_dynamic_pregame_field=False)
        == "not_applicable"
    )
    assert (
        assess_pregame_safety_from_temporal_availability("postgame_only", is_dynamic_pregame_field=False)
        == "not_applicable"
    )
