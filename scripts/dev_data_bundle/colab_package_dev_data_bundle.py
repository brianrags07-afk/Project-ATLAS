"""
Colab packaging script for the ATLAS development-data bundle.

Run this script **inside a Colab runtime with Google Drive mounted** at
``/content/drive/MyDrive/Project_Atlas`` (or an equivalent local mount for
testing against fixture data). It builds a versioned, checksum-verified
bundle of exactly the real production artifacts listed in
``atlas_reference/dev_data_bundle_required_artifacts.json`` -- nothing more.

What this script does
----------------------

1. Reads the required-artifacts allowlist.
2. For every artifact, resolves its real path under the project root,
   verifies the file exists, and fails clearly if a required
   (catalog-confirmed) artifact is missing.
3. Measures the artifact's real row count, column count, and byte size, and
   spot-checks that its declared primary-key columns exist and are unique
   at the declared grain.
4. Copies only the required files into a staging directory (never an entire
   directory tree), preserving their relative ``data/...`` path.
5. Writes a manifest conforming to
   ``atlas_reference/manifests/dev_data_bundle_manifest.schema.json``.
6. Compresses the staging directory into a single ``tar.gz`` archive.
7. Splits the archive into parts strictly smaller than 2 GiB if needed.
8. Writes a self-contained ``release_manifest.json`` (including archive and
   part checksums) suitable for upload as a GitHub Release asset alongside
   the archive part(s).

What this script deliberately does NOT do
-------------------------------------------

- It never reads, embeds, or logs credentials of any kind.
- It never copies a raw directory tree; every file it stages is an explicit,
  individually named entry from the required-artifacts allowlist.
- It never fabricates a value: if a file cannot be found or measured, the
  script raises and stops instead of writing a placeholder.

Usage
-----

    python scripts/dev_data_bundle/colab_package_dev_data_bundle.py \\
        --bundle-version 1.0.0 \\
        --project-root /content/drive/MyDrive/Project_Atlas \\
        --output-dir /content/atlas_dev_data_bundle_build
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev_data_bundle.manifest import (  # noqa: E402
    DEFAULT_MAX_PART_SIZE_BYTES,
    artifact_candidate_paths,
    load_required_artifacts,
    sha256_of_bytes,
    sha256_of_file,
    validate_manifest_or_raise,
)

ENGINE_VERSION = "1.0.0"
BUNDLE_NAME = "atlas-dev-data-bundle"
DEFAULT_SOURCE_REPOSITORY = "brianrags07-afk/Project-ATLAS"


class PackagingError(RuntimeError):
    """Raised when a required artifact cannot be found, read, or verified."""


def resolve_artifact_path(
    project_root: Path,
    artifact: dict[str, Any],
) -> Path:
    checked: list[Path] = []

    for candidate in artifact_candidate_paths(artifact):
        candidate_path = project_root / candidate

        if candidate_path.exists():
            return candidate_path

        checked.append(candidate_path)

    checked_display = "\n".join(f"  - {path}" for path in checked)

    raise PackagingError(
        f"Required artifact '{artifact['artifact_id']}' was not found under "
        f"{project_root}. Checked:\n{checked_display}\n"
        "This artifact must be built or copied to one of the above paths "
        "before packaging can continue. Refusing to fabricate a "
        "substitute."
    )


def measure_parquet(path: Path) -> tuple[int, int]:
    """Return (row_count, column_count) for a parquet file without loading
    the full file into memory when pyarrow is available."""

    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return (
            parquet_file.metadata.num_rows,
            parquet_file.metadata.num_columns,
        )
    except ImportError:
        import pandas as pd

        frame = pd.read_parquet(path)
        return (len(frame), len(frame.columns))


def measure_csv(path: Path) -> tuple[int, int]:
    import pandas as pd

    frame = pd.read_csv(path)
    return (len(frame), len(frame.columns))


def measure_artifact(path: Path) -> tuple[int, int]:
    if path.suffix == ".parquet":
        return measure_parquet(path)

    if path.suffix == ".csv":
        return measure_csv(path)

    raise PackagingError(
        f"Unsupported artifact file type for measurement: {path}"
    )


def load_key_columns(
    path: Path,
    primary_key: list[str],
) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            table = pq.ParquetFile(path).read(columns=primary_key)
            return table.to_pandas()
        except ImportError:
            return pd.read_parquet(path, columns=primary_key)

    return pd.read_csv(path, usecols=primary_key)


def verify_grain(
    path: Path,
    artifact: dict[str, Any],
) -> None:
    primary_key = artifact["primary_key"]

    try:
        key_frame = load_key_columns(path, primary_key)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise PackagingError(
            f"Could not read primary-key columns {primary_key} from "
            f"artifact '{artifact['artifact_id']}' at {path}: {exc}"
        ) from exc

    missing_columns = [
        column for column in primary_key if column not in key_frame.columns
    ]

    if missing_columns:
        raise PackagingError(
            f"Artifact '{artifact['artifact_id']}' at {path} is missing "
            f"declared primary-key column(s): {missing_columns}. Never "
            "invent column names; update the required-artifacts registry "
            "if the real grain differs."
        )

    duplicate_count = int(key_frame.duplicated().sum())

    if duplicate_count:
        raise PackagingError(
            f"Artifact '{artifact['artifact_id']}' at {path} is not unique "
            f"at its declared grain {primary_key}: {duplicate_count} "
            "duplicate row(s) found."
        )


def stage_artifact(
    project_root: Path,
    staging_dir: Path,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    source_path = resolve_artifact_path(project_root, artifact)

    verify_grain(source_path, artifact)

    row_count, column_count = measure_artifact(source_path)

    relative_path = source_path.relative_to(project_root)
    destination_path = staging_dir / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)

    return {
        "artifact_id": artifact["artifact_id"],
        "original_production_path": str(relative_path),
        "bundled_relative_path": str(relative_path),
        "file_size_bytes": source_path.stat().st_size,
        "row_count": row_count,
        "column_count": column_count,
        "primary_key": artifact["primary_key"],
        "sha256": sha256_of_file(source_path),
        "season": artifact["season"],
        "purpose": artifact["purpose"],
        "schema_reference": artifact.get("schema_reference"),
    }


def compress_staging_dir(
    staging_dir: Path,
    archive_path: Path,
) -> None:
    # staging_dir already contains a top-level "data/" directory (each
    # artifact is staged at its bundled_relative_path, e.g.
    # "data/example/example.parquet"). Add its children directly so the
    # archive's top-level entries are "data/...", not "data/data/...".
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in sorted(staging_dir.iterdir()):
            archive.add(child, arcname=child.name)


def split_archive(
    archive_path: Path,
    max_part_size_bytes: int,
) -> list[dict[str, Any]]:
    archive_size = archive_path.stat().st_size

    if archive_size <= max_part_size_bytes:
        return []

    parts: list[dict[str, Any]] = []

    with archive_path.open("rb") as handle:
        part_index = 0

        while True:
            chunk = handle.read(max_part_size_bytes)

            if not chunk:
                break

            part_filename = f"{archive_path.name}.part{part_index:03d}"
            part_path = archive_path.parent / part_filename

            with part_path.open("wb") as part_handle:
                part_handle.write(chunk)

            parts.append({
                "filename": part_filename,
                "part_index": part_index,
                "size_bytes": len(chunk),
                "sha256": sha256_of_bytes(chunk),
            })

            part_index += 1

    return parts


def build_manifest(
    *,
    bundle_version: str,
    project_root: Path,
    source_repository: str,
    artifacts: list[dict[str, Any]],
    archive_sha256: str,
    max_part_size_bytes: int,
    part_files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bundle_name": BUNDLE_NAME,
        "bundle_version": bundle_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "colab_project_root": str(project_root),
        "source_repository": source_repository,
        "source_commit": None,
        "packaging_engine_version": ENGINE_VERSION,
        "archive_format": "tar.gz",
        "archive_sha256": archive_sha256,
        "max_part_size_bytes": max_part_size_bytes,
        "part_files": part_files,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def package_bundle(
    *,
    project_root: Path,
    output_dir: Path,
    bundle_version: str,
    source_repository: str,
    max_part_size_bytes: int,
    required_artifacts_path: Path | None = None,
) -> dict[str, Any]:
    if not project_root.exists():
        raise PackagingError(
            f"Project root does not exist: {project_root}. This script "
            "must be run with Google Drive mounted (or an equivalent "
            "project-root fixture for testing)."
        )

    registry = load_required_artifacts(required_artifacts_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / "staging"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(parents=True)

    staged_artifacts = [
        stage_artifact(project_root, staging_dir, artifact)
        for artifact in registry["artifacts"]
    ]

    archive_path = output_dir / f"{BUNDLE_NAME}-{bundle_version}.tar.gz"
    compress_staging_dir(staging_dir, archive_path)
    archive_sha256 = sha256_of_file(archive_path)

    part_files = split_archive(archive_path, max_part_size_bytes)

    manifest = build_manifest(
        bundle_version=bundle_version,
        project_root=project_root,
        source_repository=source_repository,
        artifacts=staged_artifacts,
        archive_sha256=archive_sha256,
        max_part_size_bytes=max_part_size_bytes,
        part_files=part_files,
    )

    validate_manifest_or_raise(manifest)

    manifest_path = output_dir / "release_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    if part_files:
        archive_path.unlink()

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--project-root",
        default="/content/drive/MyDrive/Project_Atlas",
        help="Google Drive Project_Atlas root to read artifacts from.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the staged bundle, archive part(s), and "
        "release_manifest.json into.",
    )
    parser.add_argument(
        "--bundle-version",
        required=True,
        help="Semantic version for this bundle, e.g. 1.0.0.",
    )
    parser.add_argument(
        "--source-repository",
        default=DEFAULT_SOURCE_REPOSITORY,
        help="owner/repo recorded in the manifest.",
    )
    parser.add_argument(
        "--max-part-size-bytes",
        type=int,
        default=DEFAULT_MAX_PART_SIZE_BYTES,
        help="Archive parts will be split to be no larger than this many "
        "bytes (must be < 2147483648).",
    )
    parser.add_argument(
        "--required-artifacts",
        default=None,
        help="Path to the required-artifacts registry JSON. Defaults to "
        "atlas_reference/dev_data_bundle_required_artifacts.json.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.max_part_size_bytes >= 2 * 1024**3:
        print(
            "ERROR: --max-part-size-bytes must be strictly less than "
            "2147483648 (2 GiB).",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = package_bundle(
            project_root=Path(args.project_root),
            output_dir=Path(args.output_dir),
            bundle_version=args.bundle_version,
            source_repository=args.source_repository,
            max_part_size_bytes=args.max_part_size_bytes,
            required_artifacts_path=(
                Path(args.required_artifacts) if args.required_artifacts else None
            ),
        )
    except PackagingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Packaged {manifest['artifact_count']} artifact(s) into "
        f"{args.output_dir} (bundle_version={manifest['bundle_version']})."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
