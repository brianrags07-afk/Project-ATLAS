
from datetime import datetime, timezone


TRANSITION_SCHEMA_VERSION = "1.0.0"


def make_transition_review(
    transition_id,
    entity_type,
    entity_id,
    source_validation_id,
    source_evidence_id,
):
    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,

        "metadata": {
            "transition_id": transition_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "source_validation_id": source_validation_id,
            "source_evidence_id": source_evidence_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },

        "review_status": "TRANSITION_REVIEW_REQUIRED",

        "season_comparison": {},

        "change_signals": {
            "outcome_change": None,
            "run_differential_change": None,
            "offense_change": None,
            "pitching_change": None,
            "bullpen_change": None,
            "lineup_change": None,
            "roster_change": None,
            "manager_change": None,
            "park_change": None,
            "routine_change": None,
            "injury_change": None,
        },

        "supporting_evidence": [],

        "contradicting_evidence": [],

        "missing_evidence": [],

        "candidate_breakpoints": [],

        "decision": {
            "status": "PENDING",
            "identity_split_confirmed": False,
            "temporary_era_possible": False,
            "historical_evidence_invalidated": False,
            "reason": (
                "Outcome instability triggered review, but identity "
                "transition evidence has not yet been collected."
            ),
            "confidence": 0.0,
        },

        "pregame_safety": {
            "uses_only_historical_information": True,
            "future_results_used": False,
            "safe_for_walk_forward": True,
        },

        "traceability": {
            "source_validation_id": source_validation_id,
            "source_evidence_id": source_evidence_id,
            "source_games": [],
        },
    }
