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
import re
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.learning.concept_definition_freeze import file_sha256


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

# Required lineage-complete contract columns. A missing column is a
# certification failure -- it must never be treated as equivalent to
# ``False`` or otherwise "compliant" by omission.
REQUIRED_REGISTRY_COLUMNS = (
    "frozen_definition_id",
    "definition_sha256",
    "member_registry_sha256",
    "source_definition_registry_sha256",
    "source_member_registry_sha256",
    "discovery_season",
    "validation_season",
    "validation_engine_version",
    "validation_timestamp_utc",
    "prediction_weight_assigned",
    "2026_used",
    "validation_status",
)

REQUIRED_SUMMARY_COLUMNS = (
    "prediction_weights_assigned",
)

# Row-level SHA-256 lineage fields that must be present, non-null, and
# well-formed on every validation registry row.
ROW_LEVEL_SHA256_COLUMNS = (
    "definition_sha256",
    "member_registry_sha256",
    "source_definition_registry_sha256",
    "source_member_registry_sha256",
)

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


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


def _output_file_hashes(output_dir: Path) -> dict[str, str | None]:
    """SHA-256 of the exact four output artifacts inspected by this
    checker, keyed by filename. ``None`` if a file could not be hashed
    (e.g. it does not exist)."""

    hashes: dict[str, str | None] = {}

    for filename in REQUIRED_OUTPUT_FILENAMES:
        path = output_dir / filename
        try:
            hashes[filename] = file_sha256(str(path)) if path.exists() else None
        except Exception:
            hashes[filename] = None

    return hashes


def _resolve_expected_hash(
    expected_hash: str | None,
    source_path: Path | str | None,
) -> str | None:
    """Resolve the expected hash for a canonical input either from an
    explicitly provided hash or by independently hashing the file at
    ``source_path``. Never trusts hashes copied from output metadata --
    the whole point of this check is to recompute the hash from the
    actual canonical input artifact."""

    if expected_hash is not None:
        return expected_hash

    if source_path is None:
        return None

    source_path = Path(source_path)

    if not source_path.exists():
        return None

    try:
        return file_sha256(str(source_path))
    except Exception:
        return None


def certify_production_run(
    output_dir: Path,
    expected_frozen_definition_count: int | None = (
        PRODUCTION_EXPECTED_FROZEN_DEFINITION_COUNT
    ),
    frozen_definition_registry_path: Path | str | None = None,
    frozen_member_registry_path: Path | str | None = None,
    expected_source_definition_registry_sha256: str | None = None,
    expected_source_member_registry_sha256: str | None = None,
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

    ``frozen_definition_registry_path`` / ``frozen_member_registry_path``
    are the canonical, on-disk frozen registry files. When provided,
    this checker independently recomputes their SHA-256 (never trusting
    the hash recorded in the output metadata) and verifies that every
    validation row's ``source_definition_registry_sha256`` /
    ``source_member_registry_sha256`` matches. Callers that already
    know the expected hash may instead pass
    ``expected_source_definition_registry_sha256`` /
    ``expected_source_member_registry_sha256`` directly; an explicit
    hash always takes precedence over a path.

    Returns a dict with:

    - ``passed``: overall bool certification result
    - ``checks``: dict of individual named boolean checks
    - ``errors``: list of human-readable failure descriptions
    - ``missing_outputs``: list of required files that could not be found
    - ``counts``: key counts pulled from the outputs for reporting
    - ``output_hashes``: SHA-256 of each of the four inspected output
      artifacts, keyed by filename
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
            "output_hashes": _output_file_hashes(output_dir),
        }

    def check(name: str, condition: bool, failure_message: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(failure_message)

    # --- Required-column contract ------------------------------------
    # A missing required column is a hard certification failure. It
    # must never be treated as equivalent to ``False``/compliant.
    missing_registry_columns = sorted(
        column
        for column in REQUIRED_REGISTRY_COLUMNS
        if column not in registry.columns
    )
    missing_summary_columns = sorted(
        column
        for column in REQUIRED_SUMMARY_COLUMNS
        if column not in summary.columns
    )

    check(
        "required_registry_columns_present",
        not missing_registry_columns,
        "Validation registry is missing required column(s): "
        f"{missing_registry_columns}.",
    )

    check(
        "required_summary_columns_present",
        not missing_summary_columns,
        "Validation summary is missing required column(s): "
        f"{missing_summary_columns}.",
    )

    if missing_registry_columns or missing_summary_columns:
        # Every remaining check below assumes these columns exist. Fail
        # fast rather than silently substituting empty/False defaults.
        return {
            "passed": False,
            "checks": checks,
            "errors": errors,
            "missing_outputs": [],
            "counts": {},
            "output_hashes": _output_file_hashes(output_dir),
        }

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
            and registry_validation_seasons == {VALIDATION_SEASON}
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
            and registry_discovery_seasons == {DISCOVERY_SEASON}
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

    # --- Independent row-level lineage certification -----------------
    # These checks are computed directly from the validation registry
    # itself (and from independently recomputed source-file hashes),
    # rather than relying on aggregate flags reported by the engine's
    # own lineage audit.
    null_sha_counts: dict[str, int] = {}
    malformed_sha_columns: list[str] = []

    for column in ROW_LEVEL_SHA256_COLUMNS:
        series = registry[column]
        null_count = int(series.isna().sum())
        null_sha_counts[column] = null_count

        check(
            f"zero_null_{column}",
            null_count == 0,
            f"{null_count} null {column} value(s) found in the validation "
            "registry.",
        )

        non_null_values = series.dropna().astype(str)
        malformed = non_null_values[
            ~non_null_values.map(lambda value: bool(_SHA256_HEX_PATTERN.match(value)))
        ]

        if not malformed.empty:
            malformed_sha_columns.append(column)

        check(
            f"{column}_is_valid_sha256_hex",
            malformed.empty,
            f"{len(malformed)} {column} value(s) are not valid 64-character "
            "hexadecimal SHA-256 strings.",
        )

    source_definition_hashes = (
        registry["source_definition_registry_sha256"].dropna().astype(str)
    )
    source_member_hashes = (
        registry["source_member_registry_sha256"].dropna().astype(str)
    )

    check(
        "source_definition_registry_sha256_consistent_across_rows",
        source_definition_hashes.nunique() <= 1,
        "source_definition_registry_sha256 is not consistent across all "
        f"validation rows: {sorted(source_definition_hashes.unique())!r}.",
    )

    check(
        "source_member_registry_sha256_consistent_across_rows",
        source_member_hashes.nunique() <= 1,
        "source_member_registry_sha256 is not consistent across all "
        f"validation rows: {sorted(source_member_hashes.unique())!r}.",
    )

    expected_definition_hash = _resolve_expected_hash(
        expected_source_definition_registry_sha256,
        frozen_definition_registry_path,
    )
    expected_member_hash = _resolve_expected_hash(
        expected_source_member_registry_sha256,
        frozen_member_registry_path,
    )

    if expected_definition_hash is not None:
        check(
            "source_definition_registry_sha256_matches_frozen_file",
            (
                source_definition_hashes.nunique() == 1
                and source_definition_hashes.iloc[0] == expected_definition_hash
            ),
            "source_definition_registry_sha256 does not match the actual "
            "SHA-256 of the frozen concept definition registry file "
            f"(expected {expected_definition_hash!r}, found "
            f"{sorted(source_definition_hashes.unique())!r}).",
        )
    else:
        check(
            "source_definition_registry_sha256_matches_frozen_file",
            False,
            "Could not independently verify source_definition_registry_sha256: "
            "no frozen_definition_registry_path or expected hash was provided "
            "to certify_production_run.",
        )

    if expected_member_hash is not None:
        check(
            "source_member_registry_sha256_matches_frozen_file",
            (
                source_member_hashes.nunique() == 1
                and source_member_hashes.iloc[0] == expected_member_hash
            ),
            "source_member_registry_sha256 does not match the actual "
            "SHA-256 of the frozen concept member registry file "
            f"(expected {expected_member_hash!r}, found "
            f"{sorted(source_member_hashes.unique())!r}).",
        )
    else:
        check(
            "source_member_registry_sha256_matches_frozen_file",
            False,
            "Could not independently verify source_member_registry_sha256: "
            "no frozen_member_registry_path or expected hash was provided "
            "to certify_production_run.",
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
        "null_row_level_sha256_counts": null_sha_counts,
        "malformed_sha256_columns": malformed_sha_columns,
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
        "output_hashes": _output_file_hashes(output_dir),
    }

