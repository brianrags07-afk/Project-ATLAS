
from atlas.validation.validation_schema import make_validator_report
from atlas.validation.validation_utils import letter_grade


VALIDATOR_NAME = "data_quality_validator"
VALIDATOR_VERSION = "1.0.0"
SUPPORTED_ENTITY_TYPES = {"team", "pitcher", "batter", "bullpen", "lineup"}


def validate(evidence):
    quality = evidence.get("data_quality", {})

    missing_pct = quality.get("missing_pct")
    completeness = quality.get("sample_completeness")
    checks = quality.get("validation_checks", [])

    failing_checks = [
        check for check in checks
        if str(check).startswith("FAIL")
        or "INCOMPLETE" in str(check)
        or "MISSINGNESS_PRESENT" in str(check)
    ]

    if missing_pct is None or completeness is None:
        return make_validator_report(
            validator_name=VALIDATOR_NAME,
            validator_version=VALIDATOR_VERSION,
            status="UNKNOWN",
            grade="N/A",
            confidence=0.0,
            reason="Data-quality measurements are incomplete.",
            warnings=["MISSING_DATA_QUALITY_FIELDS"],
        )

    if failing_checks:
        score = 0.45
        status = "FAIL"
        reason = "One or more data-quality checks failed."
    elif missing_pct == 0 and completeness == 1.0:
        score = 1.0
        status = "PASS"
        reason = "No required-field missingness and complete traceability."
    elif missing_pct <= 0.01 and completeness >= 0.99:
        score = 0.92
        status = "PASS"
        reason = "Very high data completeness."
    elif missing_pct <= 0.05 and completeness >= 0.95:
        score = 0.78
        status = "PASS_WITH_CAUTION"
        reason = "Minor data-quality limitations are present."
    else:
        score = 0.55
        status = "LIMITED"
        reason = "Material missingness or incomplete traceability is present."

    return make_validator_report(
        validator_name=VALIDATOR_NAME,
        validator_version=VALIDATOR_VERSION,
        status=status,
        grade=letter_grade(score),
        confidence=score,
        reason=reason,
        warnings=failing_checks,
        details={
            "missing_pct": missing_pct,
            "sample_completeness": completeness,
            "validation_checks": checks,
        },
    )
