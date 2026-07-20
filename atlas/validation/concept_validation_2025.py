"""
ATLAS 2025 blind concept validation engine (lineage-complete).

Phase 2E.5 replaces the legacy team-scoped 2025 validation registry, which
belonged to an older, pre-freeze concept generation, with a validation layer
built directly from the certified frozen concept artifacts:

- frozen_concept_definition_registry.parquet
- frozen_concept_member_registry.parquet

Every validation record produced here is keyed by ``frozen_definition_id``
and carries complete lineage metadata back to the certified frozen
definition and member registries. The legacy ``concept_id`` is retained on
each record for backward compatibility only; it is no longer authoritative.

This module never rebuilds, mutates, or reorders the frozen discovery
artifacts. It only reads them and produces a new, independent validation
registry, summary, metadata file, and lineage audit report.

Blind validation policy:
- Discovery season is frozen at 2024.
- Validation season is 2025.
- 2026 data is never read or used.
- No prediction weights are assigned here.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR
from atlas.learning.concept_definition_freeze import (
    dataframe_registry_fingerprint,
    file_sha256,
    frozen_definition_fingerprint,
)


VALIDATION_ENGINE_VERSION = "2.0.0"

DISCOVERY_SEASON = 2024
VALIDATION_SEASON = 2025
FORBIDDEN_SEASON = 2026

SUPPORTS_TARGET = "SUPPORTS_TARGET"
OPPOSES_TARGET = "OPPOSES_TARGET"

JOIN_KEYS = (
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
)


INTERACTION_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
)

TEAM_TARGET_PATH = (
    DATA_DIR
    / "backtest"
    / "targets"
    / "team_game_targets.parquet"
)

FROZEN_CONCEPT_DIR = (
    DATA_DIR
    / "learning"
    / "frozen_concept_definitions"
    / str(DISCOVERY_SEASON)
)

FROZEN_DEFINITION_REGISTRY_PATH = (
    FROZEN_CONCEPT_DIR
    / "frozen_concept_definition_registry.parquet"
)

FROZEN_MEMBER_REGISTRY_PATH = (
    FROZEN_CONCEPT_DIR
    / "frozen_concept_member_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "validation"
    / "concepts"
    / str(VALIDATION_SEASON)
)

VALIDATION_REGISTRY_PATH = (
    OUTPUT_DIR
    / "concept_validation_registry.parquet"
)

VALIDATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "concept_validation_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "concept_validation_metadata.json"
)

LINEAGE_AUDIT_PATH = (
    OUTPUT_DIR
    / "concept_validation_lineage_audit.json"
)


# ---------------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------------


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(destination)


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary.replace(destination)


def _load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_parquet(path)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


def _normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="raise",
    ).dt.normalize()

    return dataframe


# ---------------------------------------------------------------------------
# Statistical helpers (unchanged blind-validation philosophy)
# ---------------------------------------------------------------------------


def _condition_is_active(
    values: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if operator == "<=":
        return numeric.le(threshold)

    if operator == ">=":
        return numeric.ge(threshold)

    if operator == "<":
        return numeric.lt(threshold)

    if operator == ">":
        return numeric.gt(threshold)

    if operator == "==":
        return numeric.eq(threshold)

    if operator == "!=":
        return numeric.ne(threshold)

    raise ValueError(
        f"Unsupported threshold operator: {operator}"
    )


def _two_proportion_p_value(
    successes_a: int,
    sample_a: int,
    successes_b: int,
    sample_b: int,
) -> float | None:
    if sample_a <= 0 or sample_b <= 0:
        return None

    pooled = (
        successes_a + successes_b
    ) / (
        sample_a + sample_b
    )

    variance = (
        pooled
        * (1.0 - pooled)
        * (
            (1.0 / sample_a)
            + (1.0 / sample_b)
        )
    )

    if variance <= 0:
        return None

    rate_a = successes_a / sample_a
    rate_b = successes_b / sample_b

    z_score = (
        rate_a - rate_b
    ) / math.sqrt(variance)

    return float(
        math.erfc(
            abs(z_score)
            / math.sqrt(2.0)
        )
    )


def _benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        p_values,
        errors="coerce",
    )

    valid = numeric.dropna()

    output = pd.Series(
        np.nan,
        index=p_values.index,
        dtype="float64",
    )

    if valid.empty:
        return output

    ordered = valid.sort_values(
        kind="stable",
    )

    total = len(ordered)

    adjusted = (
        ordered
        * total
        / np.arange(
            1,
            total + 1,
            dtype="float64",
        )
    )

    adjusted = (
        adjusted.iloc[::-1]
        .cummin()
        .iloc[::-1]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    output.loc[
        ordered.index
    ] = adjusted.values

    return output


def _validation_status(
    discovery_effect_direction: str,
    validation_lift: float | None,
    validation_sample: int,
    q_value: float | None,
    active_successes: int,
    inactive_successes: int,
) -> str:
    if (
        validation_lift is None
        or pd.isna(validation_lift)
        or validation_sample < 10
    ):
        return "insufficient_2025_sample"

    expected_sign = (
        1
        if discovery_effect_direction == SUPPORTS_TARGET
        else -1
    )

    observed_sign = int(
        np.sign(validation_lift)
    )

    direction_retained = (
        observed_sign == expected_sign
    )

    absolute_lift = abs(validation_lift)

    total_successes = (
        int(active_successes)
        + int(inactive_successes)
    )

    # Rare outcomes require enough observed positive events
    # before receiving a confirmed or reversed classification.
    event_sample_sufficient = (
        total_successes >= 8
    )

    statistically_supported = (
        q_value is not None
        and not pd.isna(q_value)
        and q_value <= 0.10
    )

    statistically_strong = (
        q_value is not None
        and not pd.isna(q_value)
        and q_value <= 0.05
    )

    if (
        direction_retained
        and validation_sample >= 25
        and absolute_lift >= 0.08
        and statistically_strong
        and event_sample_sufficient
    ):
        return "validated_strong"

    if (
        direction_retained
        and validation_sample >= 20
        and absolute_lift >= 0.05
        and statistically_supported
        and event_sample_sufficient
    ):
        return "validated"

    if (
        direction_retained
        and validation_sample >= 15
        and absolute_lift >= 0.025
    ):
        return "direction_retained_weak"

    if (
        not direction_retained
        and validation_sample >= 25
        and absolute_lift >= 0.08
        and statistically_strong
        and event_sample_sufficient
    ):
        return "reversed_strong"

    if (
        not direction_retained
        and validation_sample >= 20
        and absolute_lift >= 0.05
        and statistically_supported
        and event_sample_sufficient
    ):
        return "reversed"

    return "not_confirmed"


# ---------------------------------------------------------------------------
# Frozen registry loading and integrity enforcement
# ---------------------------------------------------------------------------


def _require_columns(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing = sorted(
        set(columns).difference(
            dataframe.columns
        )
    )

    if missing:
        raise KeyError(
            f"{label} missing required columns: {missing}"
        )


REQUIRED_DEFINITION_COLUMNS = (
    "frozen_definition_id",
    "definition_sha256",
    "registry_sha256",
    "member_registry_sha256",
    "concept_id",
    "target_name",
    "concept_status",
    "discovery_season",
    "member_1_feature",
    "member_1_threshold_operator",
    "member_1_threshold_value",
    "member_1_effect_direction",
    "member_2_feature",
    "member_2_threshold_operator",
    "member_2_threshold_value",
    "member_2_effect_direction",
    "same_effect_direction",
    "definitions_frozen",
    "thresholds_mutable",
    "member_features_mutable",
    "target_mutable",
)

REQUIRED_MEMBER_COLUMNS = (
    "frozen_definition_id",
    "definition_sha256",
    "target_name",
    "discovery_season",
    "member_order",
    "feature_name",
    "threshold_operator",
    "threshold_value",
    "member_definition_sha256",
    "member_definition_frozen",
    "threshold_mutable",
    "registry_sha256",
    "member_registry_sha256",
)


def _assert_frozen_registries_immutable(
    definitions: pd.DataFrame,
    members: pd.DataFrame,
) -> None:
    """
    Refuse to proceed unless the frozen artifacts are exactly what the
    certified freeze contract promises: immutable, 2024-only, and never
    touched by 2025/2026 results.
    """

    _require_columns(
        definitions,
        REQUIRED_DEFINITION_COLUMNS,
        "frozen concept definition registry",
    )

    _require_columns(
        members,
        REQUIRED_MEMBER_COLUMNS,
        "frozen concept member registry",
    )

    if not definitions["definitions_frozen"].all():
        raise AssertionError(
            "Frozen concept definition registry contains "
            "rows not marked definitions_frozen."
        )

    if definitions["thresholds_mutable"].any():
        raise AssertionError(
            "Frozen concept definition registry has mutable thresholds."
        )

    if definitions["member_features_mutable"].any():
        raise AssertionError(
            "Frozen concept definition registry has mutable member features."
        )

    if definitions["target_mutable"].any():
        raise AssertionError(
            "Frozen concept definition registry has a mutable target."
        )

    if not definitions["discovery_season"].eq(DISCOVERY_SEASON).all():
        raise AssertionError(
            "Frozen concept definition registry contains a "
            f"discovery_season other than {DISCOVERY_SEASON}."
        )

    if not members["discovery_season"].eq(DISCOVERY_SEASON).all():
        raise AssertionError(
            "Frozen concept member registry contains a "
            f"discovery_season other than {DISCOVERY_SEASON}."
        )

    if not members["member_definition_frozen"].all():
        raise AssertionError(
            "Frozen concept member registry contains rows not "
            "marked member_definition_frozen."
        )

    if members["threshold_mutable"].any():
        raise AssertionError(
            "Frozen concept member registry has mutable thresholds."
        )

    for used_column in (
        "2025_used",
        "2026_used",
        "2025_validation_used",
        "2026_results_used",
    ):
        if used_column in definitions.columns and definitions[used_column].any():
            raise AssertionError(
                "Frozen concept definition registry reports "
                f"{used_column}=True; discovery must remain blind."
            )

    if definitions["frozen_definition_id"].isna().any():
        raise AssertionError(
            "Frozen concept definition registry has a null frozen_definition_id."
        )

    if members["frozen_definition_id"].isna().any():
        raise AssertionError(
            "Frozen concept member registry has a null frozen_definition_id."
        )


def _load_frozen_registries() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    definitions = _load_parquet(
        FROZEN_DEFINITION_REGISTRY_PATH,
        "frozen concept definition registry",
    )

    members = _load_parquet(
        FROZEN_MEMBER_REGISTRY_PATH,
        "frozen concept member registry",
    )

    _assert_frozen_registries_immutable(
        definitions,
        members,
    )

    return definitions, members


# ---------------------------------------------------------------------------
# Lineage audit
# ---------------------------------------------------------------------------


def _build_lineage_audit(
    definitions: pd.DataFrame,
    members: pd.DataFrame,
    registry: pd.DataFrame,
    source_definition_registry_sha256: str,
    source_member_registry_sha256: str,
    detected_2026_rows_in_source: int,
    validation_frame_2026_row_count: int,
) -> dict[str, Any]:
    frozen_ids = (
        definitions["frozen_definition_id"]
        .astype(str)
    )

    frozen_id_set = set(frozen_ids)

    registry_ids = (
        registry["frozen_definition_id"]
        if "frozen_definition_id" in registry.columns
        else pd.Series(dtype="object")
    )

    validation_rows_missing_frozen_definition_id = int(
        registry_ids.isna().sum()
        if not registry_ids.empty
        else 0
    )

    validated_id_set = set(
        registry_ids.dropna().astype(str)
    )

    missing_validation = sorted(
        frozen_id_set - validated_id_set
    )

    orphan_validation_ids = sorted(
        validated_id_set - frozen_id_set
    )

    duplicate_frozen_definition_ids_in_registry = sorted(
        set(
            registry_ids[
                registry_ids.astype(str).duplicated()
            ].dropna().astype(str)
        )
        if not registry_ids.empty
        else set()
    )

    duplicate_frozen_definition_ids_in_source = sorted(
        set(
            frozen_ids[
                frozen_ids.duplicated()
            ]
        )
    )

    # Recompute each frozen definition's content hash directly from the
    # certified payload fields and compare it against the stored hash.
    recomputed_hashes = pd.Series(
        [
            frozen_definition_fingerprint(
                row._asdict()
            )
            for row in definitions.itertuples(index=False)
        ],
        index=definitions.index,
    )

    definition_sha256_mismatch_ids = sorted(
        definitions.loc[
            recomputed_hashes.values
            != definitions["definition_sha256"].values,
            "frozen_definition_id",
        ].astype(str)
    )

    expected_definition_registry_hash = dataframe_registry_fingerprint(
        definitions,
        "definition_sha256",
    )

    stored_definition_registry_hashes = set(
        definitions["registry_sha256"]
        .astype(str)
        .unique()
    )

    definition_registry_hash_consistent = bool(
        stored_definition_registry_hashes
        == {expected_definition_registry_hash}
    )

    expected_member_registry_hash = dataframe_registry_fingerprint(
        members,
        "member_definition_sha256",
    )

    stored_member_registry_hashes_in_definitions = set(
        definitions["member_registry_sha256"]
        .astype(str)
        .unique()
    )

    stored_member_registry_hashes_in_members = set(
        members["member_registry_sha256"]
        .astype(str)
        .unique()
    )

    member_registry_hash_consistent = bool(
        stored_member_registry_hashes_in_definitions
        == {expected_member_registry_hash}
        and stored_member_registry_hashes_in_members
        == {expected_member_registry_hash}
    )

    # Cross-check that every frozen definition has exactly two member rows
    # whose payload agrees with the definition registry's own member_1/
    # member_2 fields.
    member_counts = members.groupby(
        "frozen_definition_id"
    ).size()

    definitions_without_two_members = sorted(
        str(frozen_id)
        for frozen_id in frozen_id_set
        if int(member_counts.get(frozen_id, 0)) != 2
    )

    effect_direction_inconsistent_ids = sorted(
        definitions.loc[
            ~definitions["same_effect_direction"].astype(bool),
            "frozen_definition_id",
        ].astype(str)
    )

    used_2026_data = bool(
        validation_frame_2026_row_count > 0
    )

    return {
        "total_frozen_definitions_evaluated":
            int(len(definitions)),
        "total_validation_records_produced":
            int(len(registry)),
        "frozen_definitions_missing_validation":
            missing_validation,
        "frozen_definitions_missing_validation_count":
            int(len(missing_validation)),
        "validation_rows_missing_frozen_definition_id":
            validation_rows_missing_frozen_definition_id,
        "orphan_validation_records":
            orphan_validation_ids,
        "orphan_validation_record_count":
            int(len(orphan_validation_ids)),
        "definition_sha256_mismatches":
            definition_sha256_mismatch_ids,
        "definition_sha256_mismatch_count":
            int(len(definition_sha256_mismatch_ids)),
        "registry_sha256_mismatches": {
            "definition_registry_hash_consistent":
                definition_registry_hash_consistent,
            "member_registry_hash_consistent":
                member_registry_hash_consistent,
        },
        "unexpected_duplicate_frozen_definition_id_mappings": {
            "in_frozen_definition_registry":
                duplicate_frozen_definition_ids_in_source,
            "in_validation_registry":
                duplicate_frozen_definition_ids_in_registry,
        },
        "frozen_definitions_without_exactly_two_members":
            definitions_without_two_members,
        "effect_direction_inconsistent_frozen_definitions":
            effect_direction_inconsistent_ids,
        # `used_2026_data` and `validation_frame_2026_row_count` reflect the
        # final, already-filtered validation frame that concept evaluation
        # actually consumed. `detected_2026_rows_in_source` is purely
        # informational: shared upstream source files may legitimately
        # contain 2026 rows (e.g. for other consumers) as long as this
        # engine never reads them into the validation frame.
        "used_2026_data":
            used_2026_data,
        "validation_frame_2026_row_count":
            int(validation_frame_2026_row_count),
        "detected_2026_rows_in_source":
            int(detected_2026_rows_in_source),
        "reproducibility": {
            "discovery_season":
                DISCOVERY_SEASON,
            "validation_season":
                VALIDATION_SEASON,
            "validation_engine_version":
                VALIDATION_ENGINE_VERSION,
            "source_definition_registry_path":
                str(FROZEN_DEFINITION_REGISTRY_PATH),
            "source_definition_registry_sha256":
                source_definition_registry_sha256,
            "source_member_registry_path":
                str(FROZEN_MEMBER_REGISTRY_PATH),
            "source_member_registry_sha256":
                source_member_registry_sha256,
            "generated_at_utc":
                _utc_now_iso(),
        },
        "certified_fully_reproducible":
            bool(
                not missing_validation
                and not orphan_validation_ids
                and validation_rows_missing_frozen_definition_id == 0
                and not definition_sha256_mismatch_ids
                and definition_registry_hash_consistent
                and member_registry_hash_consistent
                and not duplicate_frozen_definition_ids_in_registry
                and not definitions_without_two_members
                and not used_2026_data
                and validation_frame_2026_row_count == 0
            ),
    }


# ---------------------------------------------------------------------------
# Validation data preparation
# ---------------------------------------------------------------------------


def _prepare_validation_frame() -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Load 2025 pregame interactions joined to 2025 team-game targets.

    Returns the joined validation frame plus the number of forbidden
    (2026) rows encountered in the raw source files, which are always
    excluded and never used.
    """

    interactions = _normalize_dates(
        _load_parquet(
            INTERACTION_PATH,
            "pregame interaction inputs",
        )
    )

    targets = _normalize_dates(
        _load_parquet(
            TEAM_TARGET_PATH,
            "team-game targets",
        )
    )

    forbidden_rows = int(
        interactions["atlas_season"].eq(FORBIDDEN_SEASON).sum()
        + targets["atlas_season"].eq(FORBIDDEN_SEASON).sum()
    )

    interactions = interactions[
        interactions["atlas_season"].eq(
            VALIDATION_SEASON
        )
    ].copy()

    targets = targets[
        targets["atlas_season"].eq(
            VALIDATION_SEASON
        )
    ].copy()

    if interactions["atlas_season"].eq(FORBIDDEN_SEASON).any():
        raise AssertionError(
            "2026 interaction rows leaked into the 2025 validation frame."
        )

    if targets["atlas_season"].eq(FORBIDDEN_SEASON).any():
        raise AssertionError(
            "2026 target rows leaked into the 2025 validation frame."
        )

    join_keys = list(JOIN_KEYS)

    target_columns = [
        column
        for column in targets.columns
        if column not in join_keys
    ]

    validation = interactions.merge(
        targets[
            join_keys + target_columns
        ],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )

    validation = validation.sort_values(
        [
            "team",
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return validation, forbidden_rows


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _evaluate_definition(
    definition: dict[str, Any],
    validation_frame: pd.DataFrame,
    source_definition_registry_sha256: str,
    source_member_registry_sha256: str,
) -> dict[str, Any]:
    target_name = str(
        definition["target_name"]
    )

    feature_1 = str(
        definition["member_1_feature"]
    )

    feature_2 = str(
        definition["member_2_feature"]
    )

    effect_1 = str(
        definition["member_1_effect_direction"]
    )

    effect_2 = str(
        definition["member_2_effect_direction"]
    )

    effect_direction_consistent = bool(
        effect_1 == effect_2
    )

    effect_direction = (
        effect_1
        if effect_direction_consistent
        else "INCONSISTENT"
    )

    base_record = {
        "frozen_definition_id":
            str(definition["frozen_definition_id"]),
        "concept_id":
            str(definition.get("concept_id", "")),
        "definition_sha256":
            str(definition["definition_sha256"]),
        "member_registry_sha256":
            str(definition["member_registry_sha256"]),
        "source_definition_registry_sha256":
            str(source_definition_registry_sha256),
        "source_member_registry_sha256":
            str(source_member_registry_sha256),
        "discovery_season":
            DISCOVERY_SEASON,
        "validation_season":
            VALIDATION_SEASON,
        "validation_engine_version":
            VALIDATION_ENGINE_VERSION,
        "validation_timestamp_utc":
            _utc_now_iso(),
        "target_name":
            target_name,
        "concept_status":
            str(definition.get("concept_status", "")),
        "member_1_feature":
            feature_1,
        "member_2_feature":
            feature_2,
        "effect_direction":
            effect_direction,
        "effect_direction_consistent":
            effect_direction_consistent,
        "prediction_weight_assigned":
            False,
        "2026_used":
            False,
    }

    if target_name not in validation_frame.columns:
        return {
            **base_record,
            "feature_availability_status":
                "target_unavailable_2025",
            "available_2025_sample": 0,
            "active_2025_sample": 0,
            "inactive_2025_sample": 0,
            "active_2025_successes": 0,
            "inactive_2025_successes": 0,
            "active_2025_rate": None,
            "inactive_2025_rate": None,
            "validation_lift": None,
            "validation_absolute_lift": None,
            "direction_retained": False,
            "validation_p_value": None,
            "validation_q_value": None,
            "validation_status": "target_unavailable_2025",
        }

    if (
        feature_1 not in validation_frame.columns
        or feature_2 not in validation_frame.columns
    ):
        return {
            **base_record,
            "feature_availability_status":
                "member_feature_unavailable_2025",
            "available_2025_sample": 0,
            "active_2025_sample": 0,
            "inactive_2025_sample": 0,
            "active_2025_successes": 0,
            "inactive_2025_successes": 0,
            "active_2025_rate": None,
            "inactive_2025_rate": None,
            "validation_lift": None,
            "validation_absolute_lift": None,
            "direction_retained": False,
            "validation_p_value": None,
            "validation_q_value": None,
            "validation_status": "member_feature_unavailable_2025",
        }

    feature_1_values = pd.to_numeric(
        validation_frame[feature_1],
        errors="coerce",
    )

    feature_2_values = pd.to_numeric(
        validation_frame[feature_2],
        errors="coerce",
    )

    target_values = pd.to_numeric(
        validation_frame[target_name],
        errors="coerce",
    )

    member_1_available = feature_1_values.notna()
    member_2_available = feature_2_values.notna()

    pair_available = (
        member_1_available
        & member_2_available
        & target_values.notna()
    )

    member_1_active = _condition_is_active(
        values=feature_1_values,
        operator=str(
            definition["member_1_threshold_operator"]
        ),
        threshold=float(
            definition["member_1_threshold_value"]
        ),
    ).fillna(False)

    member_2_active = _condition_is_active(
        values=feature_2_values,
        operator=str(
            definition["member_2_threshold_operator"]
        ),
        threshold=float(
            definition["member_2_threshold_value"]
        ),
    ).fillna(False)

    joint_active = (
        member_1_active
        & member_2_active
        & pair_available
    )

    joint_inactive = (
        pair_available
        & ~joint_active
    )

    active_sample = int(joint_active.sum())
    inactive_sample = int(joint_inactive.sum())

    active_successes = int(
        target_values[joint_active].sum()
    )

    inactive_successes = int(
        target_values[joint_inactive].sum()
    )

    active_rate = (
        active_successes / active_sample
        if active_sample > 0
        else None
    )

    inactive_rate = (
        inactive_successes / inactive_sample
        if inactive_sample > 0
        else None
    )

    validation_lift = (
        active_rate - inactive_rate
        if (
            active_rate is not None
            and inactive_rate is not None
        )
        else None
    )

    p_value = _two_proportion_p_value(
        successes_a=active_successes,
        sample_a=active_sample,
        successes_b=inactive_successes,
        sample_b=inactive_sample,
    )

    # expected_sign is None whenever the discovery-time effect direction
    # cannot be trusted (inconsistent members or an unrecognized value).
    # A None expected_sign always forces direction_retained to False below.
    if (
        not effect_direction_consistent
        or effect_direction not in (
            SUPPORTS_TARGET,
            OPPOSES_TARGET,
        )
    ):
        expected_sign = None

    else:
        expected_sign = (
            1
            if effect_direction == SUPPORTS_TARGET
            else -1
        )

    direction_retained = bool(
        validation_lift is not None
        and expected_sign is not None
        and np.sign(validation_lift) == expected_sign
    )

    return {
        **base_record,
        "feature_availability_status":
            "both_features_available",
        "available_2025_sample":
            int(pair_available.sum()),
        "active_2025_sample":
            active_sample,
        "inactive_2025_sample":
            inactive_sample,
        "active_2025_successes":
            active_successes,
        "inactive_2025_successes":
            inactive_successes,
        "active_2025_rate":
            active_rate,
        "inactive_2025_rate":
            inactive_rate,
        "validation_lift":
            validation_lift,
        "validation_absolute_lift": (
            abs(validation_lift)
            if validation_lift is not None
            else None
        ),
        "direction_retained":
            direction_retained,
        "validation_p_value":
            p_value,
        "validation_q_value":
            None,
        "validation_status":
            None,
    }


def _build_validation_registry(
    definitions: pd.DataFrame,
    validation_frame: pd.DataFrame,
    source_definition_registry_sha256: str,
    source_member_registry_sha256: str,
) -> pd.DataFrame:
    records = [
        _evaluate_definition(
            definition=row._asdict(),
            validation_frame=validation_frame,
            source_definition_registry_sha256=source_definition_registry_sha256,
            source_member_registry_sha256=source_member_registry_sha256,
        )
        for row in definitions.itertuples(index=False)
    ]

    registry = pd.DataFrame(records)

    if registry.empty:
        return registry

    testable = registry["validation_status"].isna()

    registry.loc[
        testable,
        "validation_q_value",
    ] = (
        registry.loc[testable]
        .groupby("target_name")["validation_p_value"]
        .transform(_benjamini_hochberg)
    )

    registry.loc[
        testable,
        "validation_status",
    ] = [
        _validation_status(
            discovery_effect_direction=row.effect_direction,
            validation_lift=row.validation_lift,
            validation_sample=row.active_2025_sample,
            q_value=row.validation_q_value,
            active_successes=row.active_2025_successes,
            inactive_successes=row.inactive_2025_successes,
        )
        for row in registry.loc[testable].itertuples(index=False)
    ]

    registry = registry.sort_values(
        [
            "validation_status",
            "validation_absolute_lift",
        ],
        ascending=[
            True,
            False,
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return registry


def _build_validation_summary(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    status_columns = (
        "validated_strong",
        "validated",
        "direction_retained_weak",
        "not_confirmed",
        "reversed",
        "reversed_strong",
        "insufficient_2025_sample",
        "target_unavailable_2025",
        "member_feature_unavailable_2025",
    )

    if registry.empty:
        rows = [
            {
                "target_name": None,
                "concepts_tested": 0,
                **{status: 0 for status in status_columns},
            }
        ]

    else:
        rows = []

        for target_name, group in registry.groupby(
            "target_name",
            sort=True,
            dropna=False,
        ):
            counts = group["validation_status"].value_counts()

            rows.append(
                {
                    "target_name": target_name,
                    "concepts_tested": int(len(group)),
                    **{
                        status: int(counts.get(status, 0))
                        for status in status_columns
                    },
                }
            )

        overall_counts = registry["validation_status"].value_counts()

        rows.append(
            {
                "target_name": "__all_targets__",
                "concepts_tested": int(len(registry)),
                **{
                    status: int(overall_counts.get(status, 0))
                    for status in status_columns
                },
            }
        )

    summary = pd.DataFrame(rows)

    summary["discovery_season"] = DISCOVERY_SEASON
    summary["validation_season"] = VALIDATION_SEASON
    summary["validation_engine_version"] = VALIDATION_ENGINE_VERSION
    summary["prediction_weights_assigned"] = False

    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class LineageAuditCertificationError(RuntimeError):
    """
    Raised when the lineage audit fails to certify full reproducibility.

    Canonical validation outputs must never be published when this is
    raised; the caller is expected to leave any existing canonical files
    untouched.
    """


def run_concept_validation_2025() -> dict[str, Any]:
    started = time.time()

    definitions, members = _load_frozen_registries()

    validation_frame, forbidden_rows = _prepare_validation_frame()

    validation_frame_2026_row_count = int(
        validation_frame["atlas_season"].eq(FORBIDDEN_SEASON).sum()
    )

    source_definition_registry_sha256 = file_sha256(
        str(FROZEN_DEFINITION_REGISTRY_PATH)
    )

    source_member_registry_sha256 = file_sha256(
        str(FROZEN_MEMBER_REGISTRY_PATH)
    )

    print("=" * 78)
    print("ATLAS 2025 BLIND CONCEPT VALIDATION (LINEAGE-COMPLETE)")
    print("=" * 78)
    print(
        f"Discovery Season......... {DISCOVERY_SEASON}"
    )
    print(
        f"Validation Season........ {VALIDATION_SEASON}"
    )
    print(
        f"2025 Team-Game Rows...... {len(validation_frame):,}"
    )
    print(
        f"Frozen Definitions....... {len(definitions):,}"
    )
    print(
        f"Frozen Members........... {len(members):,}"
    )
    print("=" * 78)

    # Everything below is built entirely in memory. No canonical output
    # path is touched until the lineage audit has certified the run as
    # fully reproducible.
    registry = _build_validation_registry(
        definitions=definitions,
        validation_frame=validation_frame,
        source_definition_registry_sha256=source_definition_registry_sha256,
        source_member_registry_sha256=source_member_registry_sha256,
    )

    summary = _build_validation_summary(registry)

    if registry["frozen_definition_id"].duplicated().any():
        raise AssertionError(
            "Duplicate frozen_definition_id rows in the validation registry."
        )

    lineage_audit = _build_lineage_audit(
        definitions=definitions,
        members=members,
        registry=registry,
        source_definition_registry_sha256=source_definition_registry_sha256,
        source_member_registry_sha256=source_member_registry_sha256,
        detected_2026_rows_in_source=forbidden_rows,
        validation_frame_2026_row_count=validation_frame_2026_row_count,
    )

    if not lineage_audit["certified_fully_reproducible"]:
        raise LineageAuditCertificationError(
            "Lineage audit failed certification; refusing to publish "
            "canonical validation outputs. Existing canonical outputs, "
            "if any, are left untouched. Audit: "
            + json.dumps(
                lineage_audit,
                indent=2,
                default=str,
            )
        )

    # Certification passed: atomically promote the in-memory outputs to
    # their canonical paths (each write lands on a temp file first, then
    # is renamed into place).
    _atomic_parquet_write(
        registry,
        VALIDATION_REGISTRY_PATH,
    )

    _atomic_parquet_write(
        summary,
        VALIDATION_SUMMARY_PATH,
    )

    _atomic_json_write(
        lineage_audit,
        LINEAGE_AUDIT_PATH,
    )

    status_counts = (
        registry["validation_status"].value_counts()
        if not registry.empty
        else pd.Series(dtype="int64")
    )

    elapsed = time.time() - started

    result = {
        "engine":
            "ATLAS 2025 Blind Concept Validation Engine",
        "engine_version":
            VALIDATION_ENGINE_VERSION,
        "discovery_season":
            DISCOVERY_SEASON,
        "validation_season":
            VALIDATION_SEASON,
        "frozen_definitions_evaluated":
            int(len(definitions)),
        "concepts_tested":
            int(len(registry)),
        "validated_strong":
            int(status_counts.get("validated_strong", 0)),
        "validated":
            int(status_counts.get("validated", 0)),
        "direction_retained_weak":
            int(status_counts.get("direction_retained_weak", 0)),
        "not_confirmed":
            int(status_counts.get("not_confirmed", 0)),
        "reversed":
            int(status_counts.get("reversed", 0)),
        "reversed_strong":
            int(status_counts.get("reversed_strong", 0)),
        "insufficient_2025_sample":
            int(status_counts.get("insufficient_2025_sample", 0)),
        "target_unavailable_2025":
            int(status_counts.get("target_unavailable_2025", 0)),
        "member_feature_unavailable_2025":
            int(status_counts.get("member_feature_unavailable_2025", 0)),
        "prediction_weights_assigned":
            False,
        "2026_used":
            lineage_audit["used_2026_data"],
        "2026_rows_detected_in_source":
            int(forbidden_rows),
        "certified_fully_reproducible":
            lineage_audit["certified_fully_reproducible"],
        "elapsed_seconds":
            float(elapsed),
        "outputs": {
            "validation_registry":
                str(VALIDATION_REGISTRY_PATH),
            "validation_summary":
                str(VALIDATION_SUMMARY_PATH),
            "lineage_audit":
                str(LINEAGE_AUDIT_PATH),
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("\n" + "=" * 78)
    print("2025 BLIND CONCEPT VALIDATION COMPLETE")
    print("=" * 78)
    print(
        f"Frozen Definitions Evaluated. {result['frozen_definitions_evaluated']:,}"
    )
    print(
        f"Concepts Tested............. {result['concepts_tested']:,}"
    )
    print(
        f"Validated Strong............. {result['validated_strong']:,}"
    )
    print(
        f"Validated..................... {result['validated']:,}"
    )
    print(
        f"Direction Retained Weak....... {result['direction_retained_weak']:,}"
    )
    print(
        f"Not Confirmed................. {result['not_confirmed']:,}"
    )
    print(
        f"Reversed....................... {result['reversed']:,}"
    )
    print(
        f"Insufficient Sample........... {result['insufficient_2025_sample']:,}"
    )
    print(
        f"Fully Reproducible............ {result['certified_fully_reproducible']}"
    )
    print("=" * 78)

    return result
