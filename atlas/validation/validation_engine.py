
import json
from pathlib import Path

from atlas.validation.validation_schema import make_validation_object
from atlas.validation.validation_registry import get_validators_for_evidence
from atlas.validation.validation_report import combine_validator_reports


VALIDATION_FRAMEWORK_VERSION = "1.0.0"


def validate_evidence_object(evidence):
    metadata = evidence.get("metadata", {})

    evidence_id = metadata["evidence_id"]
    question_id = metadata["question_id"]
    entity_type = metadata["entity_type"]
    entity_id = metadata["entity_id"]

    validation = make_validation_object(
        validation_id=f"VAL_{evidence_id}",
        evidence_id=evidence_id,
        question_id=question_id,
        entity_type=entity_type,
        entity_id=entity_id,
        target=metadata.get("target"),
    )

    validators = get_validators_for_evidence(evidence)
    reports = []

    for validator in validators:
        report = validator.validate(evidence)
        reports.append(report)

        validation["reviews"][
            report["validator_name"]
        ] = report

        validation["framework"]["validators_used"].append({
            "name": report["validator_name"],
            "version": report["validator_version"],
        })

    validation["combined_result"] = combine_validator_reports(reports)

    sample = evidence.get("sample", {})
    quality = evidence.get("quality", {})

    validation["validity_scope"]["season"] = sample.get("seasons")
    validation["validity_scope"]["team"] = (
        entity_id if entity_type == "team" else None
    )
    validation["validity_scope"]["active_from"] = sample.get("date_start")
    validation["validity_scope"]["active_to"] = sample.get("date_end")

    validation["transferability"] = {
        "current_context_match": None,
        "discount_required": True,
        "historical_only": False,
        "reason": (
            "Entity-local evidence only. Cross-entity transferability "
            "has not been validated."
        ),
    }

    validation["transition_review"]["split_required"] = (
        validation["combined_result"]["status"] == "SPLIT_REQUIRED"
    )

    validation["traceability"]["source_evidence_id"] = evidence_id
    validation["traceability"]["source_game_count"] = (
        evidence
        .get("traceability", {})
        .get("source_game_count")
    )
    validation["traceability"]["season_stability"] = (
        quality.get("stability")
    )

    return validation


def run_validation_engine(
    evidence_path,
    output_path=None,
):
    evidence_path = Path(evidence_path)

    if output_path is None:
        output_path = (
            evidence_path.parent.parent
            / "validation"
            / "team_moneyline_validation.json"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(evidence_path) as file:
        evidence_objects = json.load(file)

    validation_objects = [
        validate_evidence_object(evidence)
        for evidence in evidence_objects
    ]

    with open(output_path, "w") as file:
        json.dump(validation_objects, file, indent=2)

    status_counts = {}

    for validation in validation_objects:
        status = validation["combined_result"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("=" * 60)
    print("ATLAS VALIDATION FRAMEWORK")
    print("=" * 60)
    print(f"Evidence Reviewed  : {len(evidence_objects)}")
    print(f"Validation Objects : {len(validation_objects)}")
    print(f"Status Counts      : {status_counts}")
    print(f"Saved To           : {output_path}")
    print("=" * 60)

    return {
        "framework": "ATLAS Validation Framework",
        "framework_version": VALIDATION_FRAMEWORK_VERSION,
        "evidence_reviewed": len(evidence_objects),
        "validation_objects": len(validation_objects),
        "status_counts": status_counts,
        "output_path": str(output_path),
    }
