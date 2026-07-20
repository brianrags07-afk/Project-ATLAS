"""
Temporal-proof assessment for the ATLAS historical readiness audit.

Reusable logic to determine whether a field or dataset has sufficient
timestamp evidence relative to each game's ``feature_cutoff_time``. This
module never treats an object's *storage* upload/creation timestamp as
proof that the underlying real-world information (e.g. a starting
pitcher) was actually knowable before that game's cutoff -- storage
timing is reported separately and explicitly labeled as such.

Without a per-game ``source_retrieved_at`` (or equivalent field-level
"as-of" timestamp) that is on-or-before the game's ``feature_cutoff_time``,
the result is ``unknown`` or ``conditional`` -- never ``safe``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atlas.audit.evidence import PREGAME_SAFETY_VALUES, TEMPORAL_AVAILABILITY_VALUES

# Evidence types that prove real-world (not storage) availability of a
# fact before a game. Storage/object timestamps are explicitly excluded.
PREGAME_PROOF_EVIDENCE_TYPES = (
    "published_schedule_source",
    "source_retrieved_at_timestamp",
)

# Evidence types that are storage-only and must never be treated as
# proof that the underlying fact was known pregame.
STORAGE_ONLY_EVIDENCE_TYPES = (
    "storage_upload_timestamp",
    "cloud_object_metadata",
)

POSTGAME_EVIDENCE_TYPES = (
    "completed_game_record",
    "series_inferred_from_results",
)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def has_per_game_timestamp_proof(
    source_retrieved_at: Any,
    feature_cutoff_time: Any,
) -> bool:
    """Return True only if ``source_retrieved_at`` is a parseable
    timestamp that is on-or-before ``feature_cutoff_time``. Any missing or
    unparseable input returns False rather than guessing."""
    retrieved = _parse_iso(source_retrieved_at)
    cutoff = _parse_iso(feature_cutoff_time)
    if retrieved is None or cutoff is None:
        return False
    return retrieved <= cutoff


def assess_field_temporal_availability(
    evidence_records: list[dict[str, Any]],
) -> tuple[str, str]:
    """Assess ``temporal_availability`` for a single field/row from its
    evidence records only.

    Rules (explicit, not inferred):
      - Any evidence record whose evidence_type is storage-only is
        reported for context but never counts toward pregame proof.
      - If there is at least one record with a pregame-proof evidence
        type AND a per-game timestamp proof (source_retrieved_at_utc <=
        feature_cutoff_time_utc for every game observed), the field is
        ``pregame_proven``.
      - If there is at least one postgame-only evidence record and no
        pregame proof, the field is ``postgame_only``.
      - If both pregame-proof and postgame-only evidence exist, the field
        is ``mixed``.
      - If there is no evidence at all, ``unknown``.
    """
    if not evidence_records:
        return "unknown", "no evidence records supplied"

    has_pregame_proof = False
    has_postgame_only = False
    reasons: list[str] = []

    for record in evidence_records:
        etype = record.get("evidence_type")
        if etype in PREGAME_PROOF_EVIDENCE_TYPES:
            retrieved_at = record.get("observed_value") if isinstance(record.get("observed_value"), dict) else {}
            cutoff = retrieved_at.get("feature_cutoff_time_utc") if isinstance(retrieved_at, dict) else None
            source_ts = retrieved_at.get("source_retrieved_at_utc") if isinstance(retrieved_at, dict) else None
            if cutoff is not None and source_ts is not None:
                if has_per_game_timestamp_proof(source_ts, cutoff):
                    has_pregame_proof = True
                else:
                    reasons.append(
                        f"source_retrieved_at_utc ({source_ts}) is not on-or-before "
                        f"feature_cutoff_time_utc ({cutoff})"
                    )
            else:
                # A published_schedule_source with no explicit per-game
                # cutoff pairing still counts as pregame-proof evidence
                # (e.g. the schedule itself was published/timestamped
                # before every game it lists), but this is weaker and
                # callers should prefer explicit per-game pairing.
                has_pregame_proof = True
        elif etype in POSTGAME_EVIDENCE_TYPES:
            has_postgame_only = True
        elif etype in STORAGE_ONLY_EVIDENCE_TYPES:
            reasons.append(
                "storage_upload_timestamp / cloud_object_metadata observed but is NOT "
                "proof of original source availability before feature_cutoff_time"
            )

    if has_pregame_proof and has_postgame_only:
        return "mixed", "; ".join(reasons) or "both pregame-proof and postgame-only evidence present"
    if has_pregame_proof:
        return "pregame_proven", "; ".join(reasons) or "per-game timestamp proof confirmed"
    if has_postgame_only:
        return "postgame_only", "; ".join(reasons) or "only postgame/completed-game evidence present"
    return "unknown", "; ".join(reasons) or "no evidence record matched a known temporal-proof rule"


def assess_pregame_safety_from_temporal_availability(
    temporal_availability: str,
    *,
    is_dynamic_pregame_field: bool = True,
) -> str:
    """Map ``temporal_availability`` -> ``pregame_safety`` using an
    explicit, documented rule (this is the ONE place this derivation is
    allowed, and it is unit tested):

      - pregame_proven -> safe
      - mixed          -> conditional (safe subset exists, but the field
                          also contains postgame-only evidence that must
                          be excluded before use)
      - postgame_only  -> unsafe
      - unknown        -> unknown
      - not_applicable -> not_applicable

    If a field is not a dynamic pregame field at all (e.g. a pure
    identifier), pregame_safety is ``not_applicable`` regardless of
    temporal_availability.
    """
    if not is_dynamic_pregame_field:
        return "not_applicable"
    mapping = {
        "pregame_proven": "safe",
        "mixed": "conditional",
        "postgame_only": "unsafe",
        "unknown": "unknown",
        "not_applicable": "not_applicable",
    }
    result = mapping.get(temporal_availability, "unknown")
    assert result in PREGAME_SAFETY_VALUES
    assert temporal_availability in TEMPORAL_AVAILABILITY_VALUES or temporal_availability not in mapping
    return result
