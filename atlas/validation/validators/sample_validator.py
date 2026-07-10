
from atlas.validation.validation_schema import make_validator_report
from atlas.validation.validation_utils import clamp, letter_grade


VALIDATOR_NAME = "sample_validator"
VALIDATOR_VERSION = "1.0.0"
SUPPORTED_ENTITY_TYPES = {"team", "pitcher", "batter", "bullpen", "lineup"}


def validate(evidence):
    games = evidence.get("sample", {}).get("games")

    if games is None:
        return make_validator_report(
            validator_name=VALIDATOR_NAME,
            validator_version=VALIDATOR_VERSION,
            status="UNKNOWN",
            grade="N/A",
            confidence=0.0,
            reason="Evidence object does not contain a game sample.",
            warnings=["MISSING_GAME_SAMPLE"],
        )

    games = int(games)

    if games >= 300:
        score = 1.0
        status = "PASS"
        reason = f"Large entity-specific sample: {games} games."
    elif games >= 150:
        score = 0.90
        status = "PASS"
        reason = f"Strong entity-specific sample: {games} games."
    elif games >= 75:
        score = 0.78
        status = "PASS_WITH_CAUTION"
        reason = f"Moderate sample: {games} games."
    elif games >= 30:
        score = 0.62
        status = "LIMITED"
        reason = f"Limited sample: {games} games."
    else:
        score = clamp(games / 30)
        status = "INSUFFICIENT"
        reason = f"Insufficient sample: {games} games."

    return make_validator_report(
        validator_name=VALIDATOR_NAME,
        validator_version=VALIDATOR_VERSION,
        status=status,
        grade=letter_grade(score),
        confidence=score,
        reason=reason,
        details={
            "games": games,
            "entity_specific": True,
            "league_transfer_assumed": False,
        },
    )
