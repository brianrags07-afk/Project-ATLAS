"""
Explicit schedule-source assessment for the ATLAS historical readiness
audit.

Encodes the ATLAS no-leakage rule for schedules and series context:

  - Completed games appearing in ``master_game_database`` prove that a
    game *happened* (data_presence), but never prove that a *published,
    pregame* schedule existed or was retrievable before the game. Without
    a timestamped published-schedule source, provenance_status and
    pregame_safety for ``published_schedule`` / ``published_series_context``
    remain ``unknown``/``unsafe`` -- never ``complete``/``safe``.
  - A timestamped published-schedule source (retrieved/published before
    the game's scheduled start) makes both provenance_status and
    pregame_safety verifiable as ``verified``/``safe``.
  - Series boundaries/length inferred from completed historical game
    results (e.g. counting how many games a team actually played against
    an opponent) are postgame-only and must never be treated as
    pregame-safe, no matter how complete the historical record is.
"""

from __future__ import annotations

from typing import Any

from atlas.audit.evidence import make_evidence
from atlas.audit.temporal_proof import assess_field_temporal_availability


def assess_schedule_source(evidence_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess a schedule/series-context row from its evidence records.

    Returns a dict with ``provenance_status``, ``temporal_availability``,
    ``pregame_safety``, and ``reason``.
    """
    has_completed_game_only = any(
        r.get("evidence_type") == "completed_game_record" for r in evidence_records
    )
    has_published_source = any(
        r.get("evidence_type") == "published_schedule_source" for r in evidence_records
    )
    has_series_inferred = any(
        r.get("evidence_type") == "series_inferred_from_results" for r in evidence_records
    )

    temporal_availability, reason = assess_field_temporal_availability(evidence_records)

    if has_published_source:
        provenance_status = "verified"
        pregame_safety = "safe" if temporal_availability in ("pregame_proven", "mixed") else "unknown"
    elif has_series_inferred:
        provenance_status = "missing"
        pregame_safety = "unsafe"
    elif has_completed_game_only:
        provenance_status = "unknown"
        pregame_safety = "unsafe"
    else:
        provenance_status = "unknown"
        pregame_safety = "unknown"

    return {
        "provenance_status": provenance_status,
        "temporal_availability": temporal_availability,
        "pregame_safety": pregame_safety,
        "reason": reason,
    }


def evidence_from_completed_games(dataset_name: str, season: int, row_count: int) -> dict[str, Any]:
    """Build the evidence record for 'this season has completed-game rows
    in master_game_database', which is data-presence evidence only -- it
    is intentionally NOT published-schedule provenance evidence."""
    return make_evidence(
        "completed_game_record",
        source=dataset_name,
        season=season,
        observed_value={"row_count": row_count},
        confidence="observed",
        limitation=(
            "Completed games in a normalized/master dataset prove the game happened; "
            "they never prove a published, pregame schedule existed or was retrievable "
            "before the game, and must not be used to mark published_schedule complete "
            "or pregame-safe."
        ),
    )


def evidence_from_published_schedule_source(
    dataset_name: str,
    season: int,
    *,
    source_retrieved_at_utc: str | None,
    feature_cutoff_time_utc: str | None,
) -> dict[str, Any]:
    """Build the evidence record for a genuinely timestamped published
    schedule source (e.g. a schedule feed with a retrieval timestamp
    proven earlier than the games it lists)."""
    return make_evidence(
        "published_schedule_source",
        source=dataset_name,
        season=season,
        observed_value={
            "source_retrieved_at_utc": source_retrieved_at_utc,
            "feature_cutoff_time_utc": feature_cutoff_time_utc,
        },
        confidence="observed" if source_retrieved_at_utc else "unknown",
        limitation=None if source_retrieved_at_utc else "no source_retrieved_at_utc timestamp available",
    )


def evidence_from_series_inferred_from_results(dataset_name: str, season: int) -> dict[str, Any]:
    """Build the evidence record for series length/boundaries inferred
    from completed game history only -- explicitly postgame-only and
    never pregame-safe."""
    return make_evidence(
        "series_inferred_from_results",
        source=dataset_name,
        season=season,
        observed_value=None,
        confidence="heuristic",
        limitation=(
            "Series length/boundaries derived by counting completed games is a postgame "
            "fact. It must never authorize pregame prediction of series length/context."
        ),
    )
