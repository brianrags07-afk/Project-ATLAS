"""
Immutable concept-definition freeze utilities for Project ATLAS.

The freeze converts governed 2024 discovery concepts into deterministic,
content-addressed definitions that may be applied blindly to later seasons.

No prediction weights, probabilities or predictions are created here.
"""

from __future__ import annotations

from typing import Any, Final
import hashlib
import json
import math

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"
FREEZE_SCHEMA_VERSION: Final[str] = "1.0.0"


DEFINITION_FIELDS: Final[tuple[str, ...]] = (
    "discovery_season",
    "target_name",
    "concept_status",
    "broad_domain_pair",
    "member_1_feature",
    "member_1_threshold_operator",
    "member_1_threshold_value",
    "member_1_semantic_classification",
    "member_1_governance_action",
    "member_1_transformation_family_root",
    "member_1_base_metric_root",
    "member_2_feature",
    "member_2_threshold_operator",
    "member_2_threshold_value",
    "member_2_semantic_classification",
    "member_2_governance_action",
    "member_2_transformation_family_root",
    "member_2_base_metric_root",
    "semantic_family_pair_key",
    "base_metric_pair_key",
)


ALLOWED_OPERATORS: Final[set[str]] = {
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
}


ALLOWED_MEMBER_ACTION: Final[str] = (
    "KEEP_SEMANTICALLY_VALID"
)


ALLOWED_CONCEPT_STATUS: Final[str] = (
    "SEMANTICALLY_VALID_FREEZE_CANDIDATE"
)


def normalize_scalar(
    value: Any,
) -> Any:
    """
    Convert numpy/pandas scalars into deterministic JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        float,
    ):
        if math.isnan(
            value
        ):
            return None

        if math.isinf(
            value
        ):
            raise ValueError(
                "Infinite values are not valid frozen-definition fields."
            )

        return float(
            format(
                value,
                ".15g",
            )
        )

    if isinstance(
        value,
        bool,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        int,
    ):
        return int(
            value
        )

    return str(
        value
    )


def canonical_payload(
    record: dict[str, Any],
    fields: tuple[str, ...] = DEFINITION_FIELDS,
) -> dict[str, Any]:
    """
    Return a deterministically ordered definition payload.
    """

    return {
        field:
            normalize_scalar(
                record.get(
                    field
                )
            )
        for field in fields
    }


def canonical_json(
    record: dict[str, Any],
    fields: tuple[str, ...] = DEFINITION_FIELDS,
) -> str:
    """
    Serialize the frozen definition deterministically.
    """

    return json.dumps(
        canonical_payload(
            record,
            fields=fields,
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def frozen_definition_fingerprint(
    record: dict[str, Any],
) -> str:
    return sha256_text(
        canonical_json(
            record
        )
    )


def frozen_definition_id(
    record: dict[str, Any],
) -> str:
    target = str(
        record[
            "target_name"
        ]
    )

    digest = frozen_definition_fingerprint(
        record
    )

    return (
        f"{target}"
        f"__frozen_2024__"
        f"{digest[:24]}"
    )


def file_sha256(
    path: str,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as source:
        while True:
            chunk = source.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def dataframe_registry_fingerprint(
    dataframe: pd.DataFrame,
    fingerprint_column: str = "definition_sha256",
) -> str:
    values = (
        dataframe[
            fingerprint_column
        ]
        .astype(
            str
        )
        .sort_values(
            kind="stable"
        )
        .tolist()
    )

    payload = "\n".join(
        values
    )

    return sha256_text(
        payload
    )


def validate_freeze_source(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a row-level validation audit for the proposed frozen source.
    """

    required_columns = {
        "concept_id",
        "target_name",
        "concept_status",
        "broad_domain_pair",
        "member_1_feature",
        "member_1_threshold_operator",
        "member_1_threshold_value",
        "member_1_semantic_classification",
        "member_1_governance_action",
        "member_1_transformation_family_root",
        "member_1_base_metric_root",
        "member_2_feature",
        "member_2_threshold_operator",
        "member_2_threshold_value",
        "member_2_semantic_classification",
        "member_2_governance_action",
        "member_2_transformation_family_root",
        "member_2_base_metric_root",
        "semantic_family_pair_key",
        "base_metric_pair_key",
        "concept_semantic_status",
    }

    missing_columns = sorted(
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Freeze source missing required columns: "
            f"{missing_columns}"
        )

    records = []

    for row in dataframe.itertuples(
        index=False
    ):
        record = row._asdict()

        member_1_operator = str(
            record[
                "member_1_threshold_operator"
            ]
        )

        member_2_operator = str(
            record[
                "member_2_threshold_operator"
            ]
        )

        member_1_threshold = pd.to_numeric(
            pd.Series(
                [
                    record[
                        "member_1_threshold_value"
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        member_2_threshold = pd.to_numeric(
            pd.Series(
                [
                    record[
                        "member_2_threshold_value"
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        member_1_action = str(
            record[
                "member_1_governance_action"
            ]
        )

        member_2_action = str(
            record[
                "member_2_governance_action"
            ]
        )

        concept_semantic_status = str(
            record[
                "concept_semantic_status"
            ]
        )

        records.append({
            "concept_id":
                record[
                    "concept_id"
                ],

            "target_name":
                record[
                    "target_name"
                ],

            "member_1_feature_present":
                bool(
                    str(
                        record[
                            "member_1_feature"
                        ]
                    ).strip()
                ),

            "member_2_feature_present":
                bool(
                    str(
                        record[
                            "member_2_feature"
                        ]
                    ).strip()
                ),

            "member_1_operator_valid":
                member_1_operator
                in ALLOWED_OPERATORS,

            "member_2_operator_valid":
                member_2_operator
                in ALLOWED_OPERATORS,

            "member_1_threshold_valid":
                bool(
                    pd.notna(
                        member_1_threshold
                    )
                    and np.isfinite(
                        member_1_threshold
                    )
                ),

            "member_2_threshold_valid":
                bool(
                    pd.notna(
                        member_2_threshold
                    )
                    and np.isfinite(
                        member_2_threshold
                    )
                ),

            "member_1_action_valid":
                member_1_action
                == ALLOWED_MEMBER_ACTION,

            "member_2_action_valid":
                member_2_action
                == ALLOWED_MEMBER_ACTION,

            "concept_semantic_status_valid":
                concept_semantic_status
                == ALLOWED_CONCEPT_STATUS,

            "members_distinct":
                (
                    str(
                        record[
                            "member_1_feature"
                        ]
                    ),
                    member_1_operator,
                    normalize_scalar(
                        member_1_threshold
                    ),
                )
                != (
                    str(
                        record[
                            "member_2_feature"
                        ]
                    ),
                    member_2_operator,
                    normalize_scalar(
                        member_2_threshold
                    ),
                ),

            "same_base_metric_false":
                str(
                    record[
                        "member_1_base_metric_root"
                    ]
                )
                != str(
                    record[
                        "member_2_base_metric_root"
                    ]
                ),
        })

    audit = pd.DataFrame(
        records
    )

    validation_columns = [
        column
        for column in audit.columns
        if column not in {
            "concept_id",
            "target_name",
        }
    ]

    audit[
        "row_valid_for_freeze"
    ] = audit[
        validation_columns
    ].all(
        axis=1
    )

    return audit


def build_frozen_registry(
    dataframe: pd.DataFrame,
    discovery_season: int,
) -> pd.DataFrame:
    """
    Create deterministic immutable definition rows.
    """

    output = dataframe.copy()

    output[
        "discovery_season"
    ] = int(
        discovery_season
    )

    records = output.to_dict(
        orient="records"
    )

    definition_json = [
        canonical_json(
            record
        )
        for record in records
    ]

    definition_sha256 = [
        sha256_text(
            value
        )
        for value in definition_json
    ]

    output[
        "frozen_definition_id"
    ] = [
        frozen_definition_id(
            record
        )
        for record in records
    ]

    output[
        "definition_sha256"
    ] = definition_sha256

    output[
        "definition_payload_json"
    ] = definition_json

    output[
        "definition_fields_frozen"
    ] = True

    output[
        "definitions_frozen"
    ] = True

    output[
        "thresholds_mutable"
    ] = False

    output[
        "member_features_mutable"
    ] = False

    output[
        "target_mutable"
    ] = False

    output[
        "prediction_weight_assigned"
    ] = False

    output[
        "prediction_created"
    ] = False

    output[
        "2025_validation_used"
    ] = False

    output[
        "2026_results_used"
    ] = False

    output[
        "freeze_engine_version"
    ] = ENGINE_VERSION

    output[
        "freeze_schema_version"
    ] = FREEZE_SCHEMA_VERSION

    return output


def build_frozen_member_registry(
    frozen_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one immutable row per frozen concept member.
    """

    rows = []

    for concept in frozen_registry.itertuples(
        index=False
    ):
        record = concept._asdict()

        for member_order in (
            1,
            2,
        ):
            member_payload = {
                "frozen_definition_id":
                    record[
                        "frozen_definition_id"
                    ],

                "definition_sha256":
                    record[
                        "definition_sha256"
                    ],

                "target_name":
                    record[
                        "target_name"
                    ],

                "discovery_season":
                    record[
                        "discovery_season"
                    ],

                "member_order":
                    member_order,

                "feature_name":
                    record[
                        f"member_{member_order}_feature"
                    ],

                "threshold_operator":
                    record[
                        f"member_{member_order}_threshold_operator"
                    ],

                "threshold_value":
                    normalize_scalar(
                        record[
                            f"member_{member_order}_threshold_value"
                        ]
                    ),

                "semantic_classification":
                    record[
                        f"member_{member_order}_semantic_classification"
                    ],

                "governance_action":
                    record[
                        f"member_{member_order}_governance_action"
                    ],

                "transformation_family_root":
                    record[
                        f"member_{member_order}_transformation_family_root"
                    ],

                "base_metric_root":
                    record[
                        f"member_{member_order}_base_metric_root"
                    ],
            }

            member_json = json.dumps(
                member_payload,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
                ensure_ascii=True,
                allow_nan=False,
            )

            member_payload[
                "member_definition_sha256"
            ] = sha256_text(
                member_json
            )

            member_payload[
                "member_definition_payload_json"
            ] = member_json

            member_payload[
                "member_definition_frozen"
            ] = True

            member_payload[
                "threshold_mutable"
            ] = False

            rows.append(
                member_payload
            )

    return pd.DataFrame(
        rows
    )
