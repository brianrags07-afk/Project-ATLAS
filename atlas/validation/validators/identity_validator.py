
from atlas.validation.validation_schema import make_validator_report


VALIDATOR_NAME = "identity_validator"
VALIDATOR_VERSION = "0.1.0"
SUPPORTED_ENTITY_TYPES = {"team", "pitcher", "batter", "bullpen", "lineup"}


def validate(evidence):
    return make_validator_report(
        validator_name=VALIDATOR_NAME,
        validator_version=VALIDATOR_VERSION,
        status="UNKNOWN",
        grade="PENDING",
        confidence=0.0,
        reason=(
            "Identity-era, roster, injury, role, routine, and transition "
            "data are not yet attached to this evidence object."
        ),
        warnings=["IDENTITY_TRANSITION_REVIEW_PENDING"],
        details={
            "entity_specific": True,
            "league_generalization_allowed": False,
            "split_required": None,
            "current_identity_match": None,
        },
    )
