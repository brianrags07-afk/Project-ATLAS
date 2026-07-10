
from datetime import datetime


QUESTION_SCHEMA_VERSION = "1.0.0"


def make_question(
    question_id,
    subject_type,
    subject_scope,
    target,
    outcome,
    question_text,
    contexts=None,
    priority="tier_1",
    status="unanswered",
):
    return {
        "schema_version": QUESTION_SCHEMA_VERSION,
        "question_id": question_id,
        "subject_type": subject_type,
        "subject_scope": subject_scope,
        "target": target,
        "outcome": outcome,
        "question_text": question_text,
        "contexts": contexts or [],
        "priority": priority,
        "status": status,
        "evidence_status": "not_collected",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
