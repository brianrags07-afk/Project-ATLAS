
from datetime import datetime


VALIDATION_SCHEMA_VERSION = "1.0.0"


VALIDATION_STATUSES = {
    "VALID_CURRENT",
    "VALID_WITH_DISCOUNT",
    "VALID_HISTORICAL_ONLY",
    "SPLIT_REQUIRED",
    "INVALID_FOR_CURRENT_CONTEXT",
    "INSUFFICIENT_EVIDENCE",
    "TRANSITION_CANDIDATE",
    "UNKNOWN",
}


def make_validator_report(
    validator_name,
    validator_version,
    status,
    grade,
    confidence,
    reason,
    warnings=None,
    details=None,
):
    return {
        "validator_name": validator_name,
        "validator_version": validator_version,
        "status": status,
        "grade": grade,
        "confidence": confidence,
        "reason": reason,
        "warnings": warnings or [],
        "details": details or {},
    }


def make_validation_object(
    validation_id,
    evidence_id,
    question_id,
    entity_type,
    entity_id,
    target=None,
):
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,

        "metadata": {
            "validation_id": validation_id,
            "evidence_id": evidence_id,
            "question_id": question_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "target": target,
            "created_at": datetime.utcnow().isoformat(),
        },

        "framework": {
            "framework_name": "ATLAS Validation Framework",
            "framework_version": "1.0.0",
            "validators_used": [],
        },

        "reviews": {},

        "combined_result": {
            "status": "UNKNOWN",
            "recommendation": "HOLD",
            "validation_confidence": None,
            "reasons_accepted": [],
            "reasons_rejected": [],
            "warnings": [],
        },

        "validity_scope": {
            "season": None,
            "team": None,
            "park": None,
            "role": None,
            "roster_version": None,
            "lineup_version": None,
            "bullpen_version": None,
            "routine_version": None,
            "identity_version": None,
            "active_from": None,
            "active_to": None,
        },

        "transition_review": {
            "candidate": False,
            "confirmed": False,
            "split_required": False,
            "events_considered": [],
            "notes": [],
        },

        "transferability": {
            "current_context_match": None,
            "discount_required": None,
            "historical_only": None,
            "reason": None,
        },

        "traceability": {
            "source_evidence_id": evidence_id,
        },
    }
