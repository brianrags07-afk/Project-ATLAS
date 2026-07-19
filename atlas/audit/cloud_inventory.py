"""
Cloud Storage inventory helpers for the ATLAS historical readiness audit.

This module is strictly **read-only**: it only parses the output of
``gcloud storage objects list --format=json`` (or an equivalent JSON
listing) and never issues any delete/overwrite/rename/move/upload
command against Cloud Storage.

The listing call itself is made by the calling workflow/CLI via
``gcloud storage objects list ... --format=json``; this module only
normalizes and reports on the already-retrieved JSON so it can be unit
tested without any live GCS credentials.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fields we attempt to surface for every object. Any field GCS does not
# return for a given object is reported as null/empty rather than guessed.
OBJECT_FIELDS = (
    "full_path",
    "size",
    "content_type",
    "time_created",
    "updated",
    "generation",
    "metageneration",
    "md5_hash",
    "crc32c",
)


def list_bucket_objects_json(bucket: str, timeout_seconds: int = 300) -> list[dict[str, Any]]:
    """Invoke a read-only ``gcloud storage objects list`` and return the
    parsed JSON. Raises ``RuntimeError`` with a clear message on any
    authentication/listing failure. This is the only function in this
    module that talks to Cloud Storage, and it never mutates anything."""
    bucket = bucket.rstrip("/")
    cmd = [
        "gcloud",
        "storage",
        "objects",
        "list",
        f"{bucket}/**",
        "--format=json",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gcloud CLI not found; cannot list Cloud Storage objects read-only."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out listing bucket '{bucket}'.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Cloud Storage objects (authentication or permission "
            f"error most likely). stderr: {exc.stderr}"
        ) from exc

    try:
        return json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse gcloud storage objects list output as JSON: {exc}"
        ) from exc


def normalize_object_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single raw ``gcloud storage objects list --format=json``
    record into the fixed field set this audit reports on. Missing fields
    are reported as ``None`` -- never fabricated."""
    metadata = raw.get("metadata", raw)
    full_path = raw.get("url") or raw.get("name") or metadata.get("name") or metadata.get("id")
    return {
        "full_path": full_path,
        "size": _coerce_int(metadata.get("size", raw.get("size"))),
        "content_type": metadata.get("contentType", raw.get("content_type")),
        "time_created": metadata.get("timeCreated", raw.get("time_created")),
        "updated": metadata.get("updated", raw.get("updated")),
        "generation": metadata.get("generation", raw.get("generation")),
        "metageneration": metadata.get("metageneration", raw.get("metageneration")),
        "md5_hash": metadata.get("md5Hash", raw.get("md5_hash")),
        "crc32c": metadata.get("crc32c", raw.get("crc32c")),
    }


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_cloud_inventory(bucket: str, raw_objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the normalized cloud inventory dict from already-retrieved raw
    listing JSON. Pure function -- takes no network action, so it is safe
    and deterministic to unit test."""
    records = [normalize_object_record(obj) for obj in raw_objects]
    known_master_files = {
        "data/master/master_game_database.parquet",
        "data/master/master_pitch_database.parquet",
        "data/master/master_game_database_metadata.json",
        "data/master/team_game_state.parquet",
    }
    found_master_files = {
        r["full_path"].split(bucket.rstrip("/") + "/", 1)[-1]
        for r in records
        if r["full_path"]
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "object_count": len(records),
        "total_size_bytes": sum(r["size"] or 0 for r in records),
        "objects": records,
        "known_master_files_expected": sorted(known_master_files),
        "known_master_files_found": sorted(found_master_files & known_master_files),
        "known_master_files_missing": sorted(known_master_files - found_master_files),
    }


def write_cloud_inventory(inventory: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cloud_object_inventory.json"
    csv_path = output_dir / "cloud_object_inventory.csv"

    json_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OBJECT_FIELDS))
        writer.writeheader()
        for record in inventory["objects"]:
            writer.writerow({field: record.get(field) for field in OBJECT_FIELDS})

    return json_path, csv_path
