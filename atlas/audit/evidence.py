"""
Structured evidence contract for the ATLAS historical readiness audit.

The redesigned audit never collapses different evidence questions into a
single status. Every claim made anywhere in the audit (coverage matrix,
readiness decisions, provenance reports) must be backed by one or more
``EvidenceRecord`` entries built by this module. A dimension is only ever
set to a "positive" value (``present``, ``complete``, ``verified``,
``pregame_proven``, ``safe``) when a matching evidence record exists;
absence of evidence always yields ``unknown``, never a guess.

This module also defines the five independent evidence dimensions and the
allowed values for each, plus the confidence vocabulary used everywhere a
classification is heuristic rather than directly observed.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Independent evidence dimensions. These are NEVER derived from one
# another implicitly. Any rule that maps one dimension's value into
# another must be an explicit, named, tested function (see
# coverage_matrix.py / readiness.py / schedule_source_assessment.py).
# --------------------------------------------------------------------------

DATA_PRESENCE_VALUES = ("present", "partial", "missing", "unknown")
SOURCE_COMPLETENESS_VALUES = ("complete", "partial", "incomplete", "unknown", "not_applicable")
PROVENANCE_STATUS_VALUES = ("verified", "partial", "missing", "unknown")
TEMPORAL_AVAILABILITY_VALUES = (
    "pregame_proven",
    "postgame_only",
    "mixed",
    "unknown",
    "not_applicable",
)
PREGAME_SAFETY_VALUES = ("safe", "unsafe", "conditional", "unknown", "not_applicable")

DIMENSION_VALUES = {
    "data_presence": DATA_PRESENCE_VALUES,
    "source_completeness": SOURCE_COMPLETENESS_VALUES,
    "provenance_status": PROVENANCE_STATUS_VALUES,
    "temporal_availability": TEMPORAL_AVAILABILITY_VALUES,
    "pregame_safety": PREGAME_SAFETY_VALUES,
}

# Confidence for any heuristic classification. "observed" means the value
# was read directly from data/metadata with no inference; "heuristic" means
# a pattern/keyword rule was used; "unknown" means confidence itself cannot
# be established.
CONFIDENCE_VALUES = ("observed", "heuristic", "unknown")

EVIDENCE_TYPES = (
    "column_presence",
    "row_presence",
    "completed_game_record",
    "published_schedule_source",
    "series_inferred_from_results",
    "cloud_object_metadata",
    "storage_upload_timestamp",
    "source_retrieved_at_timestamp",
    "pipeline_manifest",
    "source_hash",
    "code_commit_lineage",
    "schema_fingerprint",
    "schema_compatibility_report",
    "repository_module",
    "null_value_preserved",
    "leakage_guard_result",
    "other",
)


def unknown_dimensions() -> dict[str, str]:
    """Return the default all-``unknown`` state for the five dimensions.
    Callers must overwrite a field only when they have a matching
    ``EvidenceRecord``."""
    return {
        "data_presence": "unknown",
        "source_completeness": "not_applicable",
        "provenance_status": "unknown",
        "temporal_availability": "unknown",
        "pregame_safety": "unknown",
    }


def make_evidence(
    evidence_type: str,
    source: str,
    *,
    path_or_object: str | None = None,
    field_or_column: str | None = None,
    season: int | None = None,
    observed_value: Any = None,
    confidence: str = "unknown",
    limitation: str | None = None,
) -> dict[str, Any]:
    """Build one structured evidence record. Raises ``ValueError`` for an
    unknown ``evidence_type``/``confidence`` so bad evidence can never be
    silently created."""
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"unknown evidence_type: {evidence_type!r}")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"unknown confidence: {confidence!r}")
    return {
        "evidence_type": evidence_type,
        "source": source,
        "path_or_object": path_or_object,
        "field_or_column": field_or_column,
        "season": season,
        "observed_value": observed_value,
        "confidence": confidence,
        "limitation": limitation,
    }


def validate_dimension_value(dimension: str, value: str) -> str:
    """Return ``value`` if it is legal for ``dimension``, otherwise the
    dimension's ``unknown``-equivalent value. Never raises for bad input
    from a heuristic classifier -- instead it downgrades to unknown so a
    coding bug degrades safely to 'we don't know' rather than a false
    positive."""
    allowed = DIMENSION_VALUES.get(dimension)
    if allowed is None:
        raise ValueError(f"unknown dimension: {dimension!r}")
    if value in allowed:
        return value
    return "not_applicable" if "not_applicable" in allowed else "unknown"
