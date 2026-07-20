"""
Tests for atlas/audit/evidence.py: the structured evidence contract.
"""

from __future__ import annotations

import pytest

from atlas.audit.evidence import (
    DIMENSION_VALUES,
    make_evidence,
    unknown_dimensions,
    validate_dimension_value,
)


def test_unknown_dimensions_defaults_are_all_unknown_or_not_applicable():
    dims = unknown_dimensions()
    assert dims["data_presence"] == "unknown"
    assert dims["provenance_status"] == "unknown"
    assert dims["temporal_availability"] == "unknown"
    assert dims["pregame_safety"] == "unknown"
    for key, value in dims.items():
        assert value in DIMENSION_VALUES[key]


def test_make_evidence_rejects_unknown_evidence_type():
    with pytest.raises(ValueError):
        make_evidence("not_a_real_type", source="x")


def test_make_evidence_rejects_unknown_confidence():
    with pytest.raises(ValueError):
        make_evidence("column_presence", source="x", confidence="very_sure")


def test_make_evidence_builds_full_record():
    record = make_evidence(
        "column_presence",
        source="master_game_database",
        path_or_object="data/master/master_game_database.parquet",
        field_or_column="home_score",
        season=2024,
        observed_value={"row_count": 100},
        confidence="observed",
        limitation=None,
    )
    assert record["evidence_type"] == "column_presence"
    assert record["season"] == 2024
    assert record["confidence"] == "observed"


def test_validate_dimension_value_downgrades_bad_value_to_unknown():
    assert validate_dimension_value("data_presence", "totally_present") == "unknown"
    assert validate_dimension_value("pregame_safety", "safe") == "safe"


def test_validate_dimension_value_downgrades_to_not_applicable_when_available():
    assert validate_dimension_value("source_completeness", "bogus") == "not_applicable"


def test_validate_dimension_value_rejects_unknown_dimension_name():
    with pytest.raises(ValueError):
        validate_dimension_value("not_a_dimension", "x")
