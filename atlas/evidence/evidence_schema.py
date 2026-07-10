
from datetime import datetime

EVIDENCE_SCHEMA_VERSION = "1.0.0"


def make_evidence_object(evidence_id, question_id, entity_type, entity_id, context):
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "metadata": {
            "evidence_id": evidence_id,
            "question_id": question_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "context": context,
            "created_at": datetime.utcnow().isoformat(),
        },
        "sample": {},
        "measurements": {},
        "quality": {
            "confidence": None,
            "stability": None,
            "recency": None,
        },
        "traceability": {
            "source_games": [],
        },
        "data_quality": {
            "missing_pct": None,
            "sample_completeness": None,
            "validation_checks": [],
        },
    }
