"""
Tests for atlas/audit/provenance.py: merged Cloud Storage + dataset
provenance records.
"""

from __future__ import annotations

from atlas.audit.provenance import build_all_dataset_provenance, build_dataset_provenance


def _profile():
    return {
        "cloud_path": "data/master/master_game_database.parquet",
        "row_count": 100,
        "likely_primary_key": ["game_pk"],
        "duplicate_key_count": 0,
        "schema_fingerprint": "fp-1",
        "seasons_present": ["2024"],
        "data_layer": "normalized_master",
    }


def test_missing_cloud_inventory_yields_unknown_provenance():
    record = build_dataset_provenance("master_game_database", _profile(), cloud_inventory=None)
    assert record["provenance_status"] == "unknown"
    assert record["md5_hash"] is None


def test_no_matching_cloud_object_yields_missing_provenance():
    cloud_inventory = {"objects": []}
    record = build_dataset_provenance("master_game_database", _profile(), cloud_inventory=cloud_inventory)
    assert record["provenance_status"] == "missing"


def test_hash_only_yields_partial_provenance():
    cloud_inventory = {
        "objects": [{"full_path": "gs://bucket/data/master/master_game_database.parquet", "md5_hash": "abc"}]
    }
    record = build_dataset_provenance("master_game_database", _profile(), cloud_inventory=cloud_inventory)
    assert record["provenance_status"] == "partial"
    assert record["md5_hash"] == "abc"


def test_manifest_only_yields_partial_provenance():
    record = build_dataset_provenance(
        "master_game_database", _profile(), cloud_inventory={"objects": []}, manifest_linkage="manifest-1"
    )
    assert record["provenance_status"] == "partial"


def test_hash_and_manifest_yields_verified_provenance():
    cloud_inventory = {
        "objects": [{"full_path": "gs://bucket/data/master/master_game_database.parquet", "md5_hash": "abc"}]
    }
    record = build_dataset_provenance(
        "master_game_database", _profile(), cloud_inventory=cloud_inventory, manifest_linkage="manifest-1"
    )
    assert record["provenance_status"] == "verified"


def test_storage_timestamp_note_always_present():
    record = build_dataset_provenance("master_game_database", _profile(), cloud_inventory=None)
    assert "NOT proof" in record["storage_timestamp_note"]


def test_build_all_dataset_provenance_covers_every_profile():
    profiles = {"master_game_database": _profile(), "master_pitch_database": _profile()}
    provenance = build_all_dataset_provenance(profiles, cloud_inventory=None)
    assert set(provenance.keys()) == {"master_game_database", "master_pitch_database"}
