
from atlas.validation.validators import (
    sample_validator,
    season_validator,
    data_quality_validator,
    identity_validator,
    transfer_validator,
)


VALIDATION_REGISTRY_VERSION = "1.0.0"


REGISTERED_VALIDATORS = [
    sample_validator,
    season_validator,
    data_quality_validator,
    identity_validator,
    transfer_validator,
]


def get_validators_for_evidence(evidence):
    entity_type = (
        evidence
        .get("metadata", {})
        .get("entity_type")
    )

    validators = []

    for validator in REGISTERED_VALIDATORS:
        supported = getattr(
            validator,
            "SUPPORTED_ENTITY_TYPES",
            set(),
        )

        if entity_type in supported:
            validators.append(validator)

    return validators
