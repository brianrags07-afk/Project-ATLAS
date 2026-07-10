
from atlas.identity_transition.transition_schema import (
    make_transition_review,
)
from atlas.identity_transition.transition_utils import (
    strongest_season_shift,
)


def build_team_transition_review(
    validation_object,
    evidence_object,
):
    validation_metadata = validation_object["metadata"]
    evidence_metadata = evidence_object["metadata"]

    team = evidence_metadata["entity_id"]
    validation_id = validation_metadata["validation_id"]
    evidence_id = evidence_metadata["evidence_id"]

    review = make_transition_review(
        transition_id=f"TRANSITION_TEAM_{team}",
        entity_type="team",
        entity_id=team,
        source_validation_id=validation_id,
        source_evidence_id=evidence_id,
    )

    season_breakdown = (
        evidence_object
        .get("measurements", {})
        .get("season_breakdown", {})
    )

    stability = (
        evidence_object
        .get("quality", {})
        .get("stability", {})
    )

    strongest_shift = strongest_season_shift(
        season_breakdown
    )

    review["season_comparison"] = {
        "season_breakdown": season_breakdown,
        "stability": stability,
        "strongest_shift": strongest_shift,
    }

    review["change_signals"]["outcome_change"] = {
        "detected": (
            strongest_shift is not None
            and strongest_shift["absolute_change"] >= 0.10
        ),
        "details": strongest_shift,
        "source": "team_win_pct",
        "classification": "candidate_signal_only",
    }

    review["candidate_breakpoints"] = []

    if strongest_shift is not None:
        review["candidate_breakpoints"].append({
            "from_season": strongest_shift["from_season"],
            "to_season": strongest_shift["to_season"],
            "trigger": "material_win_pct_change",
            "change": strongest_shift["win_pct_change"],
            "confirmed_identity_transition": False,
        })

    review["missing_evidence"] = [
        "roster continuity",
        "player availability and injuries",
        "confirmed lineup composition",
        "lineup order changes",
        "starting rotation composition",
        "bullpen composition and roles",
        "manager and coaching changes",
        "park or organization changes",
        "offensive profile changes",
        "contact-quality changes",
        "plate-discipline changes",
        "team pitching changes",
        "bullpen performance changes",
        "travel and rest routine changes",
    ]

    review["supporting_evidence"].append({
        "type": "season_outcome_variation",
        "description": (
            "Season win rates changed enough to justify transition review."
        ),
        "strength": "candidate_only",
    })

    review["decision"] = {
        "status": "INVESTIGATE",
        "identity_split_confirmed": False,
        "temporary_era_possible": True,
        "historical_evidence_invalidated": False,
        "reason": (
            "Season-level outcome variation was detected. "
            "A new identity era cannot be confirmed until roster, lineup, "
            "pitching, bullpen, injury, manager, and routine evidence are reviewed."
        ),
        "confidence": 0.25,
    }

    review["traceability"]["source_games"] = (
        evidence_object
        .get("traceability", {})
        .get("source_games", [])
    )

    return review
