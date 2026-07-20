"""
Tests for atlas/audit/cloud_inventory.py.

Uses only synthetic gcloud-style JSON fixtures -- never touches live
Cloud Storage credentials or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.audit.cloud_inventory import (
    build_cloud_inventory,
    list_bucket_objects_json,
    normalize_object_record,
    write_cloud_inventory,
)

BUCKET = "gs://atlas-mlb-data-brian-4817"

RAW_OBJECT_FIXTURE = {
    "url": f"{BUCKET}/data/master/master_game_database.parquet",
    "metadata": {
        "name": "data/master/master_game_database.parquet",
        "size": "12345",
        "contentType": "application/octet-stream",
        "timeCreated": "2026-01-01T00:00:00Z",
        "updated": "2026-01-02T00:00:00Z",
        "generation": "111",
        "metageneration": "2",
        "md5Hash": "abc123==",
        "crc32c": "def456==",
    },
}


def test_normalize_object_record_extracts_all_fields():
    record = normalize_object_record(RAW_OBJECT_FIXTURE)
    assert record["full_path"] == f"{BUCKET}/data/master/master_game_database.parquet"
    assert record["size"] == 12345
    assert record["content_type"] == "application/octet-stream"
    assert record["time_created"] == "2026-01-01T00:00:00Z"
    assert record["updated"] == "2026-01-02T00:00:00Z"
    assert record["generation"] == "111"
    assert record["metageneration"] == "2"
    assert record["md5_hash"] == "abc123=="
    assert record["crc32c"] == "def456=="


def test_normalize_object_record_missing_fields_are_none_not_fabricated():
    record = normalize_object_record({"metadata": {"name": "some/path"}})
    assert record["full_path"] == "some/path"
    assert record["size"] is None
    assert record["md5_hash"] is None


def test_build_cloud_inventory_detects_known_master_files_present():
    inventory = build_cloud_inventory(BUCKET, [RAW_OBJECT_FIXTURE])
    assert inventory["object_count"] == 1
    assert inventory["total_size_bytes"] == 12345
    assert "data/master/master_game_database.parquet" in inventory["known_master_files_found"]
    assert len(inventory["known_master_files_missing"]) == 3


def test_build_cloud_inventory_flags_all_known_master_files_missing_when_absent():
    inventory = build_cloud_inventory(BUCKET, [])
    assert inventory["object_count"] == 0
    assert inventory["known_master_files_found"] == []
    assert len(inventory["known_master_files_missing"]) == 4


def test_write_cloud_inventory_writes_json_and_csv(tmp_path: Path):
    inventory = build_cloud_inventory(BUCKET, [RAW_OBJECT_FIXTURE])
    json_path, csv_path = write_cloud_inventory(inventory, tmp_path)

    assert json_path.exists()
    assert csv_path.exists()

    loaded = json.loads(json_path.read_text())
    assert loaded["object_count"] == 1

    csv_text = csv_path.read_text()
    assert "full_path" in csv_text.splitlines()[0]
    assert "master_game_database.parquet" in csv_text


def test_list_bucket_objects_json_raises_clear_error_on_missing_gcloud(monkeypatch):
    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("gcloud not found")

    monkeypatch.setattr("subprocess.run", _raise_file_not_found)
    with pytest.raises(RuntimeError, match="gcloud CLI not found"):
        list_bucket_objects_json(BUCKET)


def test_list_bucket_objects_json_raises_clear_error_on_auth_failure(monkeypatch):
    import subprocess

    def _raise_called_process_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["gcloud"], stderr="PERMISSION_DENIED")

    monkeypatch.setattr("subprocess.run", _raise_called_process_error)
    with pytest.raises(RuntimeError, match="authentication or permission"):
        list_bucket_objects_json(BUCKET)


def test_list_bucket_objects_json_raises_clear_error_on_bad_json(monkeypatch):
    import subprocess

    class FakeResult:
        stdout = "not json"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    with pytest.raises(RuntimeError, match="Could not parse"):
        list_bucket_objects_json(BUCKET)
