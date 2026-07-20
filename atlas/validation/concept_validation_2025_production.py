"""
ATLAS Phase 2E.5A — production execution and certification runner for
the lineage-complete 2025 blind concept validation engine.

This module is a thin, production-safe wrapper around
``atlas.validation.concept_validation_2025.run_concept_validation_2025``.
It never re-implements discovery, freezing, thresholding, or historical
feature construction, and it never mutates any frozen or discovery
artifact. Its only job is to:

1. Print the canonical input paths, their SHA-256 hashes, and basic
   row/season counts before executing anything.
2. Execute the validation engine exactly once and let it enforce its
   own immutability and lineage-certification refusal rules.
3. Run the post-run production certification checker against the
   engine's outputs.
4. Emit a machine-readable execution manifest and a human-readable
   certification report.
5. Return a non-zero process exit code on any failure, and never
   overwrite a previously *certified* (passed) production manifest
   with the results of a failed run.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import atlas.validation.concept_validation_2025 as validation_module
from atlas.learning.concept_definition_freeze import file_sha256
from atlas.validation.concept_validation_2025_certification import (
    PRODUCTION_EXPECTED_FROZEN_DEFINITION_COUNT,
    certify_production_run,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST_DIR = (
    REPO_ROOT
    / "reports"
    / "validation_production_certification"
    / "2025"
)

MANIFEST_FILENAME = "concept_validation_production_manifest.json"
CERTIFICATION_REPORT_FILENAME = "CONCEPT_VALIDATION_2025_CERTIFICATION.md"


class ProductionRunFailure(RuntimeError):
    """Raised for any pre-flight, execution, or certification failure
    that should cause the production runner to exit non-zero without
    publishing/overwriting a certified manifest."""


def _git_commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_if_exists(path: Path) -> str | None:
    return file_sha256(str(path)) if path.exists() else None


def _season_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    if "atlas_season" not in dataframe.columns:
        return {}

    return {
        str(season): int(count)
        for season, count in (
            dataframe["atlas_season"].value_counts().sort_index().items()
        )
    }


def canonical_input_paths() -> dict[str, Path]:
    """Canonical, production ATLAS input artifact paths consumed by the
    2025 blind concept validation engine, taken directly from the
    frozen contract in ``concept_validation_2025`` (never re-derived or
    invented here)."""

    return {
        "frozen_concept_definition_registry": (
            validation_module.FROZEN_DEFINITION_REGISTRY_PATH
        ),
        "frozen_concept_member_registry": (
            validation_module.FROZEN_MEMBER_REGISTRY_PATH
        ),
        "pregame_interactions": validation_module.INTERACTION_PATH,
        "team_game_targets": validation_module.TEAM_TARGET_PATH,
    }


def canonical_output_paths() -> dict[str, Path]:
    return {
        "validation_registry": validation_module.VALIDATION_REGISTRY_PATH,
        "validation_summary": validation_module.VALIDATION_SUMMARY_PATH,
        "validation_metadata": validation_module.METADATA_PATH,
        "lineage_audit": validation_module.LINEAGE_AUDIT_PATH,
    }


def print_preflight_diagnostics(
    input_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[str]]:
    """
    Print all input paths, their SHA-256 hashes, and row/season counts
    before execution. Returns ``(diagnostics, errors)``. Read-only:
    never mutates any input.

    ``errors`` is non-empty (and the caller must fail fast, never
    proceeding into the validation engine) whenever:

    - a required input does not exist,
    - a required input's SHA-256 could not be calculated, or
    - a required Parquet input could not be read.
    """

    print("=" * 78)
    print("ATLAS 2025 CONCEPT VALIDATION -- PRODUCTION PRE-FLIGHT")
    print("=" * 78)

    diagnostics: dict[str, Any] = {}
    errors: list[str] = []

    for name, path in input_paths.items():
        print(f"Input [{name}]: {path}")

        exists = path.exists()
        sha256: str | None = None
        row_count = None
        season_counts: dict[str, int] = {}

        print(f"  exists...... {exists}")

        if not exists:
            message = f"Required input '{name}' does not exist: {path}"
            print(f"  PREFLIGHT FAILURE: {message}")
            errors.append(message)
        else:
            try:
                sha256 = file_sha256(str(path))
            except Exception as exc:
                message = (
                    f"Could not calculate SHA-256 for required input "
                    f"'{name}' ({path}): {exc}"
                )
                print(f"  PREFLIGHT FAILURE: {message}")
                errors.append(message)

            print(f"  sha256...... {sha256}")

            if path.suffix == ".parquet":
                try:
                    frame = pd.read_parquet(path)
                    row_count = int(len(frame))
                    season_counts = _season_counts(frame)
                except Exception as exc:
                    message = (
                        f"Could not read required Parquet input '{name}' "
                        f"({path}): {exc}"
                    )
                    print(f"  PREFLIGHT FAILURE: {message}")
                    errors.append(message)

            print(f"  row_count... {row_count}")
            print(f"  season_counts {season_counts}")

        diagnostics[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": sha256,
            "row_count": row_count,
            "season_counts": season_counts,
        }

    print("=" * 78)

    return diagnostics, errors


def _existing_manifest_is_certified(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            existing = json.load(file)
    except Exception:
        return False

    return bool(
        existing.get("certification", {}).get("passed") is True
    )


def _render_certification_report(manifest: dict[str, Any]) -> str:
    certification = manifest["certification"]
    result = "PASS" if certification["passed"] else "FAIL"

    lines = [
        "# ATLAS 2025 Concept Validation -- Production Certification",
        "",
        f"**Result: {result}**",
        "",
        f"- Execution timestamp (UTC): {manifest['execution_timestamp_utc']}",
        f"- Git commit SHA: {manifest['git_commit_sha']}",
        f"- Validation engine version: {manifest['validation_engine_version']}",
        f"- Elapsed runtime (s): {manifest['elapsed_seconds']}",
        "",
        "## Input Artifacts",
        "",
        "| Input | Path | SHA-256 | Rows |",
        "| --- | --- | --- | --- |",
    ]

    for name, info in manifest["inputs"].items():
        lines.append(
            f"| {name} | `{info['path']}` | `{info['sha256']}` | "
            f"{info['row_count']} |"
        )

    lines += [
        "",
        "## Output Artifacts",
        "",
        "| Output | Path | SHA-256 |",
        "| --- | --- | --- |",
    ]

    for name, info in manifest["outputs"].items():
        lines.append(
            f"| {name} | `{info['path']}` | `{info['sha256']}` |"
        )

    certified_output_hashes = certification.get("output_hashes", {})

    if certified_output_hashes:
        lines += [
            "",
            "## Certified Output Hashes",
            "",
            "SHA-256 of the exact four output artifacts inspected by the "
            "certification checker.",
            "",
            "| Output File | SHA-256 |",
            "| --- | --- |",
        ]
        for filename, sha256 in certified_output_hashes.items():
            lines.append(f"| {filename} | `{sha256}` |")

    lines += [
        "",
        "## Critical Counts",
        "",
        f"- Frozen definitions evaluated: "
        f"{manifest['frozen_definition_count']}",
        f"- Frozen members: {manifest['frozen_member_count']}",
        f"- Validation records produced: "
        f"{manifest['validation_record_count']}",
        "",
        "## Validation Status Counts",
        "",
    ]

    for status, count in manifest["validation_status_counts"].items():
        lines.append(f"- {status}: {count}")

    lines += [
        "",
        "## Certification Checks",
        "",
    ]

    for name, passed in certification["checks"].items():
        mark = "PASS" if passed else "FAIL"
        lines.append(f"- [{mark}] {name}")

    if certification["errors"]:
        lines += [
            "",
            "## Certification Errors",
            "",
        ]
        for error in certification["errors"]:
            lines.append(f"- {error}")

    lines += [
        "",
        "## Notes",
        "",
        "This report reflects the outputs of a single production "
        "execution. It does not certify anything beyond the four "
        "canonical validation outputs it inspects. Discovery, frozen "
        "definitions, frozen members, thresholds, targets, and "
        "historical feature construction are never modified by this "
        "runner.",
        "",
    ]

    return "\n".join(lines) + "\n"


def run_production_validation(
    manifest_dir: Path | None = None,
    expected_frozen_definition_count: int | None = (
        PRODUCTION_EXPECTED_FROZEN_DEFINITION_COUNT
    ),
) -> tuple[int, dict[str, Any]]:
    """
    Execute the production concept validation run end to end.

    Returns ``(exit_code, manifest)``. ``exit_code`` is 0 only when the
    run executed, all immutability/lineage checks in the engine
    itself passed, and the post-run production certification checker
    passed. ``manifest`` is always returned (even on failure) so
    callers can inspect what happened; it is only *persisted* to disk
    when doing so would not overwrite a previously certified manifest
    with a failed result.
    """

    manifest_dir = manifest_dir or DEFAULT_MANIFEST_DIR
    manifest_path = manifest_dir / MANIFEST_FILENAME
    report_path = manifest_dir / CERTIFICATION_REPORT_FILENAME

    started = time.time()
    execution_timestamp_utc = _utc_now_iso()
    git_commit_sha = _git_commit_sha()

    input_paths = canonical_input_paths()
    output_paths = canonical_output_paths()

    input_diagnostics, preflight_errors = print_preflight_diagnostics(
        input_paths
    )

    engine_result: dict[str, Any] | None = None
    execution_error: str | None = None

    if preflight_errors:
        # Fail fast: never call the validation engine when a required
        # input is missing, unhashable, or (for Parquet inputs)
        # unreadable.
        execution_error = "Production pre-flight failure: " + "; ".join(
            preflight_errors
        )
    else:
        try:
            engine_result = validation_module.run_concept_validation_2025()
        except validation_module.LineageAuditCertificationError as exc:
            execution_error = f"Lineage certification failure: {exc}"
        except (AssertionError, KeyError, FileNotFoundError) as exc:
            execution_error = (
                f"Frozen registry immutability / input validation failure: {exc}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            execution_error = f"Unexpected execution failure: {exc}"

    elapsed_seconds = time.time() - started

    output_diagnostics = {
        name: {
            "path": str(path),
            "sha256": _hash_if_exists(path),
        }
        for name, path in output_paths.items()
    }

    if execution_error is not None:
        certification = {
            "passed": False,
            "checks": {},
            "errors": [execution_error],
            "missing_outputs": [],
            "counts": {},
            "output_hashes": {},
        }
    else:
        certification = certify_production_run(
            output_dir=Path(output_paths["validation_registry"]).parent,
            expected_frozen_definition_count=expected_frozen_definition_count,
            frozen_definition_registry_path=(
                input_paths["frozen_concept_definition_registry"]
            ),
            frozen_member_registry_path=(
                input_paths["frozen_concept_member_registry"]
            ),
        )

    manifest = {
        "execution_timestamp_utc": execution_timestamp_utc,
        "git_commit_sha": git_commit_sha,
        "validation_engine_version": (
            validation_module.VALIDATION_ENGINE_VERSION
        ),
        "inputs": input_diagnostics,
        "outputs": output_diagnostics,
        "frozen_definition_count": (
            engine_result.get("frozen_definitions_evaluated")
            if engine_result
            else None
        ),
        "frozen_member_count": input_diagnostics.get(
            "frozen_concept_member_registry", {}
        ).get("row_count"),
        "validation_record_count": (
            engine_result.get("concepts_tested") if engine_result else None
        ),
        "validation_status_counts": certification.get("counts", {}).get(
            "validation_status_counts", {}
        ),
        "execution_error": execution_error,
        "certification": certification,
        "elapsed_seconds": elapsed_seconds,
    }

    passed = execution_error is None and certification["passed"]

    if passed:
        # Atomic write-via-rename: write to a temporary sibling file first
        # (e.g. "concept_validation_production_manifest.json.tmp"), then
        # rename it into place, so a crash mid-write can never leave a
        # partially-written canonical manifest/report on disk.
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_tmp_path = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )
        with open(manifest_tmp_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, default=str)
        manifest_tmp_path.replace(manifest_path)

        report_text = _render_certification_report(manifest)
        report_tmp_path = report_path.with_suffix(
            report_path.suffix + ".tmp"
        )
        report_tmp_path.write_text(
            report_text,
            encoding="utf-8",
        )
        report_tmp_path.replace(report_path)

        print(f"Production manifest written to: {manifest_path}")
        print(f"Production certification report written to: {report_path}")
    else:
        if _existing_manifest_is_certified(manifest_path):
            print(
                "REFUSING TO OVERWRITE an existing CERTIFIED production "
                f"manifest at {manifest_path} with a failed run's results."
            )
        else:
            manifest_dir.mkdir(parents=True, exist_ok=True)
            failed_manifest_path = manifest_dir / (
                MANIFEST_FILENAME + ".failed.json"
            )
            with open(failed_manifest_path, "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, default=str)

            failed_report_path = manifest_dir / (
                CERTIFICATION_REPORT_FILENAME.replace(
                    ".md", "_FAILED.md"
                )
            )
            failed_report_path.write_text(
                _render_certification_report(manifest),
                encoding="utf-8",
            )

            print(
                "Certification FAILED. Failure manifest/report written to: "
                f"{failed_manifest_path}, {failed_report_path}"
            )

        for error in certification.get("errors", []):
            print(f"CERTIFICATION ERROR: {error}")

        if execution_error:
            print(f"EXECUTION ERROR: {execution_error}")

    exit_code = 0 if passed else 1

    return exit_code, manifest


def main() -> int:
    exit_code, _manifest = run_production_validation()
    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
