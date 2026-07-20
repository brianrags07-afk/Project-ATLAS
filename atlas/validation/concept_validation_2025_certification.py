"""
ATLAS Phase 2E.5A — post-run production certification checker.

This module never rebuilds, mutates, or re-derives any validation
output. It only *reads* the four canonical outputs produced by
``atlas.validation.concept_validation_2025.run_concept_validation_2025``
and verifies that they satisfy the production certification contract:

- ``concept_validation_registry.parquet``
- ``concept_validation_summary.parquet``
- ``concept_validation_metadata.json``
- ``concept_validation_lineage_audit.json``

The checker is intentionally decoupled from the validation engine
itself so it can be exercised independently (including against
synthetic fixtures in tests) and can be re-run against a previously
published production output without repeating the validation run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# The canonical, certified frozen-2024 concept discovery run produced
# exactly this many frozen definitions. This is a fixed production
# expectation, not something re-derived from the data being checked.
# It must only be updated if a *new*, separately-certified frozen 2024
# concept discovery run officially replaces the current frozen
# registries under data/learning/frozen_concept_definitions/2024/.
PRODUCTION_EXPECTED_FROZEN_DEFINITION_COUNT = 2138

DISCOVERY_SEASON = 2024
VALIDATION_SEASON = 2025
FORBIDDEN_SEASON = 2026

REGISTRY_FILENAME = "concept_validation_registry.parquet"
SUMMARY_FILENAME = "concept_validation_summary.parquet"
METADATA_FILENAME = "concept_validation_metadata.json"
LINEAGE_AUDIT_FILENAME = "concept_validation_lineage_audit.json"

REQUIRED_OUTPUT_FILENAMES = (
    REGISTRY_FILENAME,
    SUMMARY_FILENAME,
    METADATA_FILENAME,
    LINEAGE_AUDIT_FILENAME,
)


class CertificationLoadError(RuntimeError):
    """Raised when a required production output is missing or unreadable."""


def _load_outputs(
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    list[str],
]:
    """
    Load the four required production outputs from ``output_dir``.

    Returns ``(registry, summary, metadata, lineage_audit, missing)``.
    ``missing`` lists any required filenames that could not be found;
    when non-empty, the corresponding return value is an empty
    placeholder (empty DataFrame / empty dict) and certification must
    fail.
    """

    missing: list[str] = []

    registry_path = output_dir / REGISTRY_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    lineage_audit_path = output_dir / LINEAGE_AUDIT_FILENAME

    if registry_path.exists():
        try:
            registry = pd.read_parquet(registry_path)
        except Exception:
            missing.append(REGISTRY_FILENAME)
            registry = pd.DataFrame()
    else:
        missing.append(REGISTRY_FILENAME)
        registry = pd.DataFrame()

    if summary_path.exists():
        try:
            summary = pd.read_parquet(summary_path)
        except Exception:
            missing.append(SUMMARY_FILENAME)
            summary = pd.DataFrame()
    else:
        missing.append(SUMMARY_FILENAME)
        summary = pd.DataFrame()

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        except Exception:
            missing.append(METADATA_FILENAME)
            metadata = {}
    else:
        missing.append(METADATA_FILENAME)
        metadata = {}

    if lineage_audit_path.exists():
        try:
            with open(lineage_audit_path, "r", encoding="utf-8") as file:
                lineage_audit = json.load(file)
        except Exception:
            missing.append(LINEAGE_AUDIT_FILENAME)
            lineage_audit = {}
    else:
        missing.append(LINEAGE_AUDIT_FILENAME)
        lineage_audit = {}

    return registry, summary, metadata, lineage_audit, missing


def certify_production_run(
    output_dir: Path,
    expected_frozen_definition_count: int | None = (
        PRODUCTION_EXPECTED_FROZEN_DEFINITION_COUNT
    ),
) -> dict[str, Any]:
    """
    Load and verify the four production validation outputs in
    ``output_dir`` against the Phase 2E.5A certification contract.

    ``expected_frozen_definition_count`` pins the exact frozen
    definition / validation record / unique id count that a genuine
    production run against the real 2024 frozen registries must
    produce (2,138). Tests exercising synthetic fixtures pass a
    different value (or ``None`` to skip the exact-count checks and
    rely on internal cross-file consistency only).

    Returns a dict with:

    - ``passed``: overall bool certification result
    - ``checks``: dict of individual named boolean checks
    - ``errors``: list of human-readable failure descriptions
    - ``missing_outputs``: list of required files that could not be found
    - ``counts``: key counts pulled from the outputs for reporting
    """

    (
        registry,
        summary,
        metadata,
        lineage_audit,
        missing_outputs,
    ) = _load_outputs(output_dir)

    errors: list[str] = []
    checks: dict[str, bool] = {}

    if missing_outputs:
        errors.append(
            "Missing required production outputs: "
            + ", ".join(sorted(missing_outputs))
        )

        return {
            "passed": False,
            "checks": checks,
            "errors": errors,
            "missing_outputs": sorted(missing_outputs),
            "counts": {},
        }

    def check(name: str, condition: bool, failure_message: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(failure_message)

    frozen_definitions_evaluated = metadata.get("frozen_definitions_evaluated")
    concepts_tested = metadata.get("concepts_tested")
    lineage_total_frozen = lineage_audit.get(
        "total_frozen_definitions_evaluated"
    )
    lineage_total_records = lineage_audit.get(
        "total_validation_records_produced"
    )

    registry_frozen_ids = (
        registry["frozen_definition_id"]
        if "frozen_definition_id" in registry.columns
        else pd.Series(dtype="object")
    )

    unique_frozen_id_count = int(
        registry_frozen_ids.dropna().astype(str).nunique()
    )

    null_frozen_id_count = int(registry_frozen_ids.isna().sum())

    duplicate_registry_rows = int(
        registry_frozen_ids.astype(str).duplicated().sum()
        if not registry_frozen_ids.empty
        else 0
    )

    duplicate_mapping = lineage_audit.get(
        "unexpected_duplicate_frozen_definition_id_mappings",
        {},
    )
    duplicate_in_source = duplicate_mapping.get(
        "in_frozen_definition_registry",
        [],
    )
    duplicate_in_registry = duplicate_mapping.get(
        "in_validation_registry",
        [],
    )

    registry_hash_flags = lineage_audit.get(
        "registry_sha256_mismatches",
        {},
    )

    reproducibility = lineage_audit.get("reproducibility", {})

    # --- Counts -----------------------------------------------------
    check(
        "frozen_definitions_evaluated_consistent",
        frozen_definitions_evaluated == lineage_total_frozen,
        "metadata.frozen_definitions_evaluated "
        f"({frozen_definitions_evaluated!r}) does not match "
        f"lineage_audit.total_frozen_definitions_evaluated "
        f"({lineage_total_frozen!r}).",
    )

    check(
        "validation_records_produced_consistent",
        (
            concepts_tested == lineage_total_records
            and concepts_tested == len(registry)
        ),
        "metadata.concepts_tested "
        f"({concepts_tested!r}), "
        "lineage_audit.total_validation_records_produced "
        f"({lineage_total_records!r}), and registry row count "
        f"({len(registry)}) must all match.",
    )

    if expected_frozen_definition_count is not None:
        check(
            "frozen_definitions_evaluated_equals_expected",
            frozen_definitions_evaluated == expected_frozen_definition_count,
            "frozen definitions evaluated "
            f"({frozen_definitions_evaluated!r}) != expected "
            f"({expected_frozen_definition_count}).",
        )

        check(
            "validation_records_produced_equals_expected",
            len(registry) == expected_frozen_definition_count,
            f"validation records produced ({len(registry)}) != expected "
            f"({expected_frozen_definition_count}).",
        )

        check(
            "unique_frozen_definition_id_equals_expected",
            unique_frozen_id_count == expected_frozen_definition_count,
            "unique frozen_definition_id values "
            f"({unique_frozen_id_count}) != expected "
            f"({expected_frozen_definition_count}).",
        )
    else:
        check(
            "unique_frozen_definition_id_equals_record_count",
            unique_frozen_id_count == len(registry),
            "unique frozen_definition_id values "
            f"({unique_frozen_id_count}) != validation record count "
            f"({len(registry)}).",
        )

    check(
        "zero_null_frozen_definition_id",
        null_frozen_id_count == 0,
        f"{null_frozen_id_count} null frozen_definition_id value(s) found.",
    )

    check(
        "zero_orphan_validation_records",
        lineage_audit.get("orphan_validation_record_count") == 0,
        "orphan_validation_record_count is "
        f"{lineage_audit.get('orphan_validation_record_count')!r}, expected 0.",
    )

    check(
        "zero_missing_frozen_definitions",
        lineage_audit.get("frozen_definitions_missing_validation_count") == 0,
        "frozen_definitions_missing_validation_count is "
        f"{lineage_audit.get('frozen_definitions_missing_validation_count')!r}"
        ", expected 0.",
    )

    check(
        "zero_definition_sha256_mismatches",
        lineage_audit.get("definition_sha256_mismatch_count") == 0,
        "definition_sha256_mismatch_count is "
        f"{lineage_audit.get('definition_sha256_mismatch_count')!r}, expected 0.",
    )

    check(
        "definition_registry_hash_consistent",
        registry_hash_flags.get("definition_registry_hash_consistent") is True,
        "definition_registry_hash_consistent is "
        f"{registry_hash_flags.get('definition_registry_hash_consistent')!r}"
        ", expected True.",
    )

    check(
        "member_registry_hash_consistent",
        registry_hash_flags.get("member_registry_hash_consistent") is True,
        "member_registry_hash_consistent is "
        f"{registry_hash_flags.get('member_registry_hash_consistent')!r}"
        ", expected True.",
    )

    check(
        "zero_unexpected_duplicate_frozen_definition_id_rows",
        (
            duplicate_registry_rows == 0
            and not duplicate_in_source
            and not duplicate_in_registry
        ),
        "duplicate frozen_definition_id rows detected "
        f"(registry duplicates={duplicate_registry_rows}, "
        f"source duplicates={duplicate_in_source}, "
        f"registry-mapping duplicates={duplicate_in_registry}).",
    )

    # --- Blind validation policy -------------------------------------
    used_2026_data = lineage_audit.get("used_2026_data")
    validation_frame_2026_row_count = lineage_audit.get(
        "validation_frame_2026_row_count"
    )

    check(
        "used_2026_data_false",
        used_2026_data is False and metadata.get("2026_used") is False,
        "used_2026_data must be False in both the lineage audit "
        f"({used_2026_data!r}) and metadata ({metadata.get('2026_used')!r}).",
    )

    check(
        "zero_2026_rows_in_validation_frame",
        validation_frame_2026_row_count == 0,
        "validation_frame_2026_row_count is "
        f"{validation_frame_2026_row_count!r}, expected 0.",
    )

    registry_validation_seasons = (
        set(registry["validation_season"].dropna().unique().tolist())
        if "validation_season" in registry.columns
        else set()
    )

    check(
        "validation_season_2025_only",
        (
            metadata.get("validation_season") == VALIDATION_SEASON
            and reproducibility.get("validation_season") == VALIDATION_SEASON
            and registry_validation_seasons in ({VALIDATION_SEASON}, set())
        ),
        "validation season must be exactly "
        f"{VALIDATION_SEASON} everywhere "
        f"(metadata={metadata.get('validation_season')!r}, "
        f"lineage_audit={reproducibility.get('validation_season')!r}, "
        f"registry={sorted(registry_validation_seasons)!r}).",
    )

    registry_discovery_seasons = (
        set(registry["discovery_season"].dropna().unique().tolist())
        if "discovery_season" in registry.columns
        else set()
    )

    check(
        "discovery_season_2024",
        (
            metadata.get("discovery_season") == DISCOVERY_SEASON
            and reproducibility.get("discovery_season") == DISCOVERY_SEASON
            and registry_discovery_seasons in ({DISCOVERY_SEASON}, set())
        ),
        "discovery season must be exactly "
        f"{DISCOVERY_SEASON} everywhere "
        f"(metadata={metadata.get('discovery_season')!r}, "
        f"lineage_audit={reproducibility.get('discovery_season')!r}, "
        f"registry={sorted(registry_discovery_seasons)!r}).",
    )

    registry_prediction_weight_assigned = (
        bool(registry["prediction_weight_assigned"].any())
        if "prediction_weight_assigned" in registry.columns
        else False
    )

    summary_prediction_weights_assigned = (
        bool(summary["prediction_weights_assigned"].any())
        if "prediction_weights_assigned" in summary.columns
        else False
    )

    check(
        "prediction_weights_assigned_false",
        (
            metadata.get("prediction_weights_assigned") is False
            and not registry_prediction_weight_assigned
            and not summary_prediction_weights_assigned
        ),
        "prediction_weights_assigned must be False everywhere "
        f"(metadata={metadata.get('prediction_weights_assigned')!r}, "
        f"registry_any={registry_prediction_weight_assigned}, "
        f"summary_any={summary_prediction_weights_assigned}).",
    )

    check(
        "certified_fully_reproducible_true",
        (
            lineage_audit.get("certified_fully_reproducible") is True
            and metadata.get("certified_fully_reproducible") is True
        ),
        "certified_fully_reproducible must be True in both the lineage "
        f"audit ({lineage_audit.get('certified_fully_reproducible')!r}) and "
        f"metadata ({metadata.get('certified_fully_reproducible')!r}).",
    )

    passed = all(checks.values())

    counts = {
        "frozen_definitions_evaluated": frozen_definitions_evaluated,
        "validation_records_produced": len(registry),
        "unique_frozen_definition_id_count": unique_frozen_id_count,
        "null_frozen_definition_id_count": null_frozen_id_count,
        "duplicate_frozen_definition_id_rows": duplicate_registry_rows,
        "orphan_validation_record_count": lineage_audit.get(
            "orphan_validation_record_count"
        ),
        "frozen_definitions_missing_validation_count": lineage_audit.get(
            "frozen_definitions_missing_validation_count"
        ),
        "definition_sha256_mismatch_count": lineage_audit.get(
            "definition_sha256_mismatch_count"
        ),
        "validation_frame_2026_row_count": validation_frame_2026_row_count,
        "validation_status_counts": (
            registry["validation_status"].value_counts().to_dict()
            if "validation_status" in registry.columns
            else {}
        ),
    }

    return {
        "passed": bool(passed),
        "checks": checks,
        "errors": errors,
        "missing_outputs": [],
        "counts": counts,
    }
