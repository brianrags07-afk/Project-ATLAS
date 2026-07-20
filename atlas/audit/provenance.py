"""
Provenance merge for the ATLAS historical readiness audit.

Combines Cloud Storage object metadata (from ``cloud_inventory``) with
dataset profiling results (from ``dataset_profile``) into one provenance
record per known dataset. Any field this audit cannot observe directly is
left ``None`` -- it is never inferred or guessed.

``provenance_status`` here is intentionally conservative:
  - ``verified`` requires BOTH a Cloud Storage content hash (md5/crc32c)
    matched to the exact object path AND an explicit manifest linkage
    (pipeline_manifest / source_manifest_id) tying that object to a
    recorded pipeline run. Neither alone is sufficient.
  - ``partial`` requires at least one of those two, but not both.
  - ``missing`` means neither the object nor any manifest reference was
    found.
  - ``unknown`` means this audit could not evaluate the object at all
    (e.g. cloud inventory unavailable).
"""

from __future__ import annotations

from typing import Any

PROVENANCE_FIELDS = (
    "gcs_path",
    "generation",
    "metageneration",
    "md5_hash",
    "crc32c",
    "size",
    "created_at",
    "updated_at",
    "season_range",
    "schema_fingerprint",
    "row_count",
    "candidate_primary_key",
    "duplicate_key_count",
    "producing_module",
    "producing_commit_or_version",
    "parent_source_objects",
    "manifest_linkage",
    "evidence_confidence",
)


def _find_cloud_object(cloud_inventory: dict[str, Any] | None, cloud_path_suffix: str) -> dict[str, Any] | None:
    if not cloud_inventory:
        return None
    for obj in cloud_inventory.get("objects", []):
        full_path = obj.get("full_path") or ""
        if full_path.endswith(cloud_path_suffix):
            return obj
    return None


def build_dataset_provenance(
    dataset_name: str,
    profile: dict[str, Any],
    cloud_inventory: dict[str, Any] | None = None,
    manifest_linkage: str | None = None,
    producing_module: str | None = None,
    producing_commit_or_version: str | None = None,
    parent_source_objects: list[str] | None = None,
) -> dict[str, Any]:
    """Build one provenance record. Never fabricates a value -- any field
    without direct evidence stays ``None``."""
    cloud_path = profile.get("cloud_path")
    cloud_object = _find_cloud_object(cloud_inventory, cloud_path) if cloud_path else None

    has_hash = bool(cloud_object and (cloud_object.get("md5_hash") or cloud_object.get("crc32c")))
    has_manifest = bool(manifest_linkage)

    if has_hash and has_manifest:
        provenance_status = "verified"
        confidence = "observed"
    elif has_hash or has_manifest:
        provenance_status = "partial"
        confidence = "observed" if has_hash else "heuristic"
    elif cloud_inventory is None:
        provenance_status = "unknown"
        confidence = "unknown"
    else:
        provenance_status = "missing"
        confidence = "unknown"

    seasons = profile.get("seasons_present") or []

    return {
        "dataset": dataset_name,
        "gcs_path": cloud_path,
        "generation": (cloud_object or {}).get("generation"),
        "metageneration": (cloud_object or {}).get("metageneration"),
        "md5_hash": (cloud_object or {}).get("md5_hash"),
        "crc32c": (cloud_object or {}).get("crc32c"),
        "size": (cloud_object or {}).get("size"),
        "created_at": (cloud_object or {}).get("time_created"),
        "updated_at": (cloud_object or {}).get("updated"),
        "season_range": seasons,
        "schema_fingerprint": profile.get("schema_fingerprint"),
        "row_count": profile.get("row_count"),
        "candidate_primary_key": profile.get("likely_primary_key"),
        "duplicate_key_count": profile.get("duplicate_key_count"),
        "producing_module": producing_module,
        "producing_commit_or_version": producing_commit_or_version,
        "parent_source_objects": parent_source_objects or [],
        "manifest_linkage": manifest_linkage,
        "data_layer": profile.get("data_layer"),
        "provenance_status": provenance_status,
        "evidence_confidence": confidence,
        "storage_timestamp_note": (
            "created_at/updated_at above are Cloud Storage object timestamps only. They are "
            "NOT proof that the underlying real-world data was available/knowable before any "
            "particular game's feature_cutoff_time."
        ),
    }


def build_all_dataset_provenance(
    dataset_profiles: dict[str, dict[str, Any]],
    cloud_inventory: dict[str, Any] | None = None,
    manifest_linkages: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    manifest_linkages = manifest_linkages or {}
    return {
        name: build_dataset_provenance(
            name, profile, cloud_inventory, manifest_linkage=manifest_linkages.get(name)
        )
        for name, profile in dataset_profiles.items()
    }
