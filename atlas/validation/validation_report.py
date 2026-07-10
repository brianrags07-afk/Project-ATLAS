
from atlas.validation.validation_utils import mean_confidence


def combine_validator_reports(reports):
    statuses = {
        report["validator_name"]: report["status"]
        for report in reports
    }

    warnings = []
    accepted = []
    rejected = []

    for report in reports:
        warnings.extend(report.get("warnings", []))

        if report["status"] in {
            "PASS",
            "PASS_WITH_CAUTION",
            "LOCAL_ONLY",
        }:
            accepted.append(report["reason"])

        if report["status"] in {
            "FAIL",
            "INSUFFICIENT",
            "SPLIT_REVIEW",
        }:
            rejected.append(report["reason"])

    confidence = mean_confidence(reports)

    if statuses.get("data_quality_validator") == "FAIL":
        status = "INVALID_FOR_CURRENT_CONTEXT"
        recommendation = "REJECT"

    elif statuses.get("sample_validator") == "INSUFFICIENT":
        status = "INSUFFICIENT_EVIDENCE"
        recommendation = "HOLD"

    elif statuses.get("season_validator") == "SPLIT_REVIEW":
        status = "SPLIT_REQUIRED"
        recommendation = "SPLIT_AND_REVALIDATE"

    elif statuses.get("identity_validator") == "UNKNOWN":
        status = "VALID_WITH_DISCOUNT"
        recommendation = "ACCEPT_WITH_IDENTITY_CAUTION"

    else:
        status = "VALID_CURRENT"
        recommendation = "ACCEPT"

    return {
        "status": status,
        "recommendation": recommendation,
        "validation_confidence": confidence,
        "reasons_accepted": accepted,
        "reasons_rejected": rejected,
        "warnings": sorted(set(warnings)),
    }
