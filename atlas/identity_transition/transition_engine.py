
import json
from pathlib import Path

from atlas.identity_transition.team_transition import (
    build_team_transition_review,
)


TRANSITION_ENGINE_VERSION = "1.0.0"


def run_identity_transition_engine(
    validation_path,
    evidence_path,
    output_path=None,
):
    validation_path = Path(validation_path)
    evidence_path = Path(evidence_path)

    if output_path is None:
        output_path = (
            evidence_path.parent.parent
            / "identity_transition"
            / "team_transition_reviews.json"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(validation_path) as file:
        validations = json.load(file)

    with open(evidence_path) as file:
        evidence_objects = json.load(file)

    evidence_by_id = {
        item["metadata"]["evidence_id"]: item
        for item in evidence_objects
    }

    reviews = []

    for validation in validations:
        status = (
            validation
            .get("combined_result", {})
            .get("status")
        )

        if status != "SPLIT_REQUIRED":
            continue

        evidence_id = (
            validation
            .get("metadata", {})
            .get("evidence_id")
        )

        evidence = evidence_by_id.get(evidence_id)

        if evidence is None:
            raise KeyError(
                f"Evidence object missing for {evidence_id}"
            )

        reviews.append(
            build_team_transition_review(
                validation_object=validation,
                evidence_object=evidence,
            )
        )

    with open(output_path, "w") as file:
        json.dump(reviews, file, indent=2)

    decision_counts = {}

    for review in reviews:
        decision = review["decision"]["status"]
        decision_counts[decision] = (
            decision_counts.get(decision, 0) + 1
        )

    print("=" * 70)
    print("ATLAS IDENTITY TRANSITION ENGINE")
    print("=" * 70)
    print(f"Validation Objects Read : {len(validations)}")
    print(f"Transition Reviews      : {len(reviews)}")
    print(f"Decision Counts         : {decision_counts}")
    print(f"Saved To                : {output_path}")
    print("=" * 70)

    return {
        "engine": "ATLAS Identity Transition Engine",
        "engine_version": TRANSITION_ENGINE_VERSION,
        "validation_objects_read": len(validations),
        "transition_reviews": len(reviews),
        "decision_counts": decision_counts,
        "output_path": str(output_path),
    }
