
from atlas.validation.validation_schema import make_validator_report
from atlas.validation.validation_utils import letter_grade


VALIDATOR_NAME = "season_validator"
VALIDATOR_VERSION = "1.0.0"
SUPPORTED_ENTITY_TYPES = {"team", "pitcher", "batter", "bullpen", "lineup"}


def validate(evidence):
    measurements = evidence.get("measurements", {})
    season_breakdown = measurements.get("season_breakdown", {})
    stability = evidence.get("quality", {}).get("stability", {})

    if not season_breakdown:
        return make_validator_report(
            validator_name=VALIDATOR_NAME,
            validator_version=VALIDATOR_VERSION,
            status="UNKNOWN",
            grade="N/A",
            confidence=0.0,
            reason="Season-separated evidence is unavailable.",
            warnings=["MISSING_SEASON_BREAKDOWN"],
        )

    seasons = sorted(season_breakdown.keys())
    stability_label = stability.get("label", "not_measured")
    stability_range = stability.get("range")

    if len(seasons) < 2:
        return make_validator_report(
            validator_name=VALIDATOR_NAME,
            validator_version=VALIDATOR_VERSION,
            status="INSUFFICIENT",
            grade="D",
            confidence=0.45,
            reason="Only one season is available.",
            warnings=["MULTI_SEASON_VALIDATION_UNAVAILABLE"],
            details={"seasons": seasons},
        )

    if stability_label == "high":
        score = 0.95
        status = "PASS"
        reason = "Entity evidence is highly stable across separated seasons."
    elif stability_label == "medium":
        score = 0.82
        status = "PASS_WITH_CAUTION"
        reason = "Entity evidence is moderately stable across separated seasons."
    elif stability_label == "low":
        score = 0.58
        status = "SPLIT_REVIEW"
        reason = "Season results vary materially; season or identity splitting may be required."
    else:
        score = 0.40
        status = "UNKNOWN"
        reason = "Season stability has not been measured."

    return make_validator_report(
        validator_name=VALIDATOR_NAME,
        validator_version=VALIDATOR_VERSION,
        status=status,
        grade=letter_grade(score),
        confidence=score,
        reason=reason,
        warnings=(
            ["POSSIBLE_SEASON_OR_IDENTITY_SHIFT"]
            if status == "SPLIT_REVIEW"
            else []
        ),
        details={
            "seasons": seasons,
            "season_breakdown": season_breakdown,
            "stability_label": stability_label,
            "stability_range": stability_range,
            "season_separation_preserved": True,
        },
    )
