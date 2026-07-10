
from atlas.validation.validation_schema import make_validator_report


VALIDATOR_NAME = "transfer_validator"
VALIDATOR_VERSION = "0.1.0"
SUPPORTED_ENTITY_TYPES = {"team", "pitcher", "batter", "bullpen", "lineup"}


def validate(evidence):
    entity_id = evidence.get("metadata", {}).get("entity_id")

    return make_validator_report(
        validator_name=VALIDATOR_NAME,
        validator_version=VALIDATOR_VERSION,
        status="LOCAL_ONLY",
        grade="PENDING",
        confidence=0.50,
        reason=(
            f"Evidence is validated only for entity {entity_id}. "
            "No transfer to other teams, players, or league-wide contexts is assumed."
        ),
        warnings=["TRANSFERABILITY_NOT_TESTED"],
        details={
            "entity_id": entity_id,
            "local_before_global": True,
            "transfer_validated": False,
            "league_weight_inherited": False,
        },
    )
