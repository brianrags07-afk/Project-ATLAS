"""
Shared manifest helpers for the ATLAS development-data bundle tooling.

This module contains no production scientific logic. It defines the
structure and lightweight validation of:

- ``atlas_reference/dev_data_bundle_required_artifacts.json`` (the allowlist
  of real production artifacts a bundle must contain), and
- the bundle manifest written by the Colab packaging script and consumed by
  the repository bootstrap script, which must conform to
  ``atlas_reference/manifests/dev_data_bundle_manifest.schema.json``.

Validation here is intentionally implemented with the standard library only
(no ``jsonschema`` dependency) so it can run unmodified inside a bare Colab
runtime or a fresh developer checkout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARTIFACTS_PATH = (
    REPO_ROOT
    / "atlas_reference"
    / "dev_data_bundle_required_artifacts.json"
)

MANIFEST_SCHEMA_PATH = (
    REPO_ROOT
    / "atlas_reference"
    / "manifests"
    / "dev_data_bundle_manifest.schema.json"
)

# Strictly less than 2 GiB (2 * 1024**3 bytes), per the task requirement
# that split parts must be smaller than 2 GiB.
DEFAULT_MAX_PART_SIZE_BYTES = 1_900_000_000

SHA256_HEX_LENGTH = 64

REQUIRED_ARTIFACT_ENTRY_KEYS = (
    "artifact_id",
    "original_production_path",
    "season",
    "purpose",
    "primary_key",
    "catalog_status",
)

REQUIRED_MANIFEST_ARTIFACT_KEYS = (
    "artifact_id",
    "original_production_path",
    "bundled_relative_path",
    "file_size_bytes",
    "row_count",
    "column_count",
    "primary_key",
    "sha256",
    "season",
    "purpose",
)

REQUIRED_MANIFEST_TOP_LEVEL_KEYS = (
    "bundle_name",
    "bundle_version",
    "created_utc",
    "colab_project_root",
    "source_repository",
    "packaging_engine_version",
    "artifact_count",
    "artifacts",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest or required-artifacts registry is malformed."""


def load_required_artifacts(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load and lightly validate the required-artifacts registry."""

    registry_path = path or REQUIRED_ARTIFACTS_PATH

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Required-artifacts registry not found: {registry_path}"
        )

    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    errors = validate_required_artifacts(registry)

    if errors:
        raise ManifestValidationError(
            "Invalid required-artifacts registry "
            f"({registry_path}):\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    return registry


def validate_required_artifacts(
    registry: dict[str, Any],
) -> list[str]:
    """Return a list of validation errors (empty if the registry is valid)."""

    errors: list[str] = []

    artifacts = registry.get("artifacts")

    if not isinstance(artifacts, list) or not artifacts:
        return ["'artifacts' must be a non-empty list."]

    seen_ids: set[str] = set()

    for index, artifact in enumerate(artifacts):
        prefix = f"artifacts[{index}]"

        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        for key in REQUIRED_ARTIFACT_ENTRY_KEYS:
            if key not in artifact:
                errors.append(f"{prefix} is missing required key '{key}'.")

        artifact_id = artifact.get("artifact_id")

        if isinstance(artifact_id, str):
            if artifact_id in seen_ids:
                errors.append(f"{prefix} duplicate artifact_id '{artifact_id}'.")
            seen_ids.add(artifact_id)

        original_path = artifact.get("original_production_path")

        if isinstance(original_path, str) and not original_path.startswith("data/"):
            errors.append(
                f"{prefix} original_production_path must start with 'data/': "
                f"{original_path!r}"
            )

        primary_key = artifact.get("primary_key")

        if primary_key is not None and (
            not isinstance(primary_key, list) or not primary_key
        ):
            errors.append(f"{prefix} primary_key must be a non-empty list.")

        catalog_status = artifact.get("catalog_status")

        if catalog_status not in (
            "confirmed_in_schema_catalog",
            "path_unconfirmed_requires_colab_verification",
        ):
            errors.append(
                f"{prefix} catalog_status has an unrecognized value: "
                f"{catalog_status!r}"
            )

    return errors


def load_manifest_schema(
    path: Path | None = None,
) -> dict[str, Any]:
    schema_path = path or MANIFEST_SCHEMA_PATH

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Manifest JSON Schema not found: {schema_path}"
        )

    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(
    manifest: dict[str, Any],
) -> list[str]:
    """
    Validate a packaged bundle manifest against the required structural
    contract. Returns a list of human-readable errors (empty if valid).

    This is a deliberately lightweight, dependency-free structural check
    (not a full JSON Schema validator). It enforces the same required
    fields documented in
    ``atlas_reference/manifests/dev_data_bundle_manifest.schema.json``.
    """

    errors: list[str] = []

    for key in REQUIRED_MANIFEST_TOP_LEVEL_KEYS:
        if key not in manifest:
            errors.append(f"manifest is missing required key '{key}'.")

    artifacts = manifest.get("artifacts")

    if not isinstance(artifacts, list):
        errors.append("manifest 'artifacts' must be a list.")
        return errors

    if manifest.get("artifact_count") != len(artifacts):
        errors.append(
            "manifest 'artifact_count' "
            f"({manifest.get('artifact_count')!r}) does not match "
            f"len(artifacts) ({len(artifacts)})."
        )

    bundle_version = manifest.get("bundle_version")

    if isinstance(bundle_version, str):
        parts = bundle_version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            errors.append(
                f"manifest 'bundle_version' must be 'MAJOR.MINOR.PATCH': "
                f"{bundle_version!r}"
            )

    for index, artifact in enumerate(artifacts):
        prefix = f"artifacts[{index}]"

        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        for key in REQUIRED_MANIFEST_ARTIFACT_KEYS:
            if key not in artifact:
                errors.append(f"{prefix} is missing required key '{key}'.")

        sha256_value = artifact.get("sha256")

        if isinstance(sha256_value, str):
            if not _is_valid_sha256_hex(sha256_value):
                errors.append(
                    f"{prefix} sha256 is not a valid 64-character hex "
                    f"digest: {sha256_value!r}"
                )

        for numeric_key in ("file_size_bytes", "row_count", "column_count"):
            value = artifact.get(numeric_key)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                errors.append(
                    f"{prefix} {numeric_key} must be a non-negative integer, "
                    f"got {value!r}."
                )

        primary_key = artifact.get("primary_key")

        if primary_key is not None and (
            not isinstance(primary_key, list) or not primary_key
        ):
            errors.append(f"{prefix} primary_key must be a non-empty list.")

    part_files = manifest.get("part_files") or []

    for index, part in enumerate(part_files):
        part_prefix = f"part_files[{index}]"

        if not isinstance(part, dict):
            errors.append(f"{part_prefix} must be an object.")
            continue

        for key in ("filename", "part_index", "size_bytes", "sha256"):
            if key not in part:
                errors.append(f"{part_prefix} is missing required key '{key}'.")

        part_sha256 = part.get("sha256")

        if isinstance(part_sha256, str) and not _is_valid_sha256_hex(part_sha256):
            errors.append(
                f"{part_prefix} sha256 is not a valid 64-character hex "
                f"digest: {part_sha256!r}"
            )

    max_part_size = manifest.get("max_part_size_bytes")

    if max_part_size is not None and max_part_size >= 2 * 1024**3:
        errors.append(
            "manifest 'max_part_size_bytes' must be strictly less than "
            f"2147483648 (2 GiB); got {max_part_size!r}."
        )

    return errors


def validate_manifest_or_raise(
    manifest: dict[str, Any],
) -> None:
    errors = validate_manifest(manifest)

    if errors:
        raise ManifestValidationError(
            "Invalid dev-data-bundle manifest:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def _is_valid_sha256_hex(value: str) -> bool:
    if len(value) != SHA256_HEX_LENGTH:
        return False

    try:
        int(value, 16)
    except ValueError:
        return False

    return True


def sha256_of_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Compute the SHA-256 digest of a file's exact bytes, streamed."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_artifact_by_id(
    registry: dict[str, Any],
    artifact_id: str,
) -> dict[str, Any] | None:
    for artifact in registry.get("artifacts", []):
        if artifact.get("artifact_id") == artifact_id:
            return artifact

    return None


def artifact_candidate_paths(
    artifact: dict[str, Any],
) -> Iterable[str]:
    """
    Yield every relative path (under the Drive project root) that should be
    checked for this artifact, in priority order. Most artifacts have a
    single known path; a few unconfirmed 2025 artifacts carry an explicit
    ``candidate_paths`` list because more than one plausible production
    path exists in the repository's own documentation.
    """

    candidates = artifact.get("candidate_paths")

    if candidates:
        yield from candidates
        return

    yield artifact["original_production_path"]
