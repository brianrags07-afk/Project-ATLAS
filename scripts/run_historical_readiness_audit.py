#!/usr/bin/env python3
"""
Entry point for the ATLAS historical readiness audit.

This script is invoked by the ``atlas-historical-readiness-audit`` GitHub
Actions workflow. It is strictly read-only with respect to Cloud Storage:

  * it lists bucket contents with ``gcloud storage objects list`` (a read
    operation),
  * it downloads the four known master datasets with ``gcloud storage cp``
    from the bucket to a local, ephemeral directory (a read operation),
  * it NEVER runs ``gcloud storage cp`` (or any other command) *to* the
    bucket, and never deletes/overwrites/renames/moves any Cloud Storage
    object.

All findings are written under ``artifacts/audits/`` for upload as GitHub
Actions artifacts only -- nothing produced by this script is uploaded back
to Cloud Storage.

Usage:
    python scripts/run_historical_readiness_audit.py \
        --bucket gs://atlas-mlb-data-brian-4817 \
        --local-data-dir data/master \
        --output-dir artifacts/audits
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas.audit import (  # noqa: E402
    cloud_inventory as cloud_inventory_mod,
    coverage_matrix as coverage_matrix_mod,
    dataset_profile as dataset_profile_mod,
    job_summary as job_summary_mod,
    readiness as readiness_mod,
    repository_inventory as repository_inventory_mod,
)

KNOWN_MASTER_FILES = (
    "master_game_database.parquet",
    "master_pitch_database.parquet",
    "master_game_database_metadata.json",
    "team_game_state.parquet",
)


class AuditError(RuntimeError):
    """Raised for any authentication, download, parsing, or validation
    error so the workflow fails clearly instead of silently continuing."""


def _fail(stage: str, message: str) -> None:
    raise AuditError(f"[{stage}] {message}")


def download_known_master_files(bucket: str, local_dir: Path) -> dict[str, Path]:
    """Read-only download (``gcloud storage cp`` FROM the bucket only) of
    the four known master datasets. Never writes to the bucket."""
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket = bucket.rstrip("/")
    downloaded: dict[str, Path] = {}
    for filename in KNOWN_MASTER_FILES:
        remote_path = f"{bucket}/data/master/{filename}"
        local_path = local_dir / filename
        cmd = ["gcloud", "storage", "cp", remote_path, str(local_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        except FileNotFoundError as exc:
            _fail("download", f"gcloud CLI not found: {exc}")
        except subprocess.TimeoutExpired as exc:
            _fail("download", f"Timed out downloading {remote_path}: {exc}")
        except subprocess.CalledProcessError as exc:
            _fail(
                "download",
                f"Failed to download {remote_path} (authentication or missing-object error). "
                f"stderr: {exc.stderr}",
            )
        if not local_path.exists():
            _fail("download", f"Expected local file not found after download: {local_path}")
        downloaded[filename] = local_path
    return downloaded


def load_profiles(downloaded: dict[str, Path]) -> tuple[dict, dict]:
    import pandas as pd

    profiles: dict = {}
    try:
        game_df = pd.read_parquet(downloaded["master_game_database.parquet"])
        profiles["master_game_database"] = dataset_profile_mod.profile_master_game_database(
            game_df, "data/master/master_game_database.parquet",
            downloaded["master_game_database.parquet"].stat().st_size,
        )
    except Exception as exc:  # noqa: BLE001
        _fail("parsing", f"Failed to parse master_game_database.parquet: {exc}")

    try:
        pitch_df = pd.read_parquet(downloaded["master_pitch_database.parquet"])
        profiles["master_pitch_database"] = dataset_profile_mod.profile_master_pitch_database(
            pitch_df, "data/master/master_pitch_database.parquet",
            downloaded["master_pitch_database.parquet"].stat().st_size,
        )
    except Exception as exc:  # noqa: BLE001
        _fail("parsing", f"Failed to parse master_pitch_database.parquet: {exc}")

    try:
        team_df = pd.read_parquet(downloaded["team_game_state.parquet"])
        profiles["team_game_state"] = dataset_profile_mod.profile_team_game_state(
            team_df, "data/master/team_game_state.parquet",
            downloaded["team_game_state.parquet"].stat().st_size,
        )
    except Exception as exc:  # noqa: BLE001
        _fail("parsing", f"Failed to parse team_game_state.parquet: {exc}")

    try:
        metadata = json.loads(downloaded["master_game_database_metadata.json"].read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _fail("parsing", f"Failed to parse master_game_database_metadata.json: {exc}")

    metadata_comparison = dataset_profile_mod.profile_metadata_json(metadata, profiles)
    return profiles, metadata_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="ATLAS historical readiness audit (read-only).")
    parser.add_argument("--bucket", default="gs://atlas-mlb-data-brian-4817")
    parser.add_argument("--local-data-dir", default="data/master")
    parser.add_argument("--output-dir", default="artifacts/audits")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    local_data_dir = REPO_ROOT / args.local_data_dir

    try:
        print("== Step 1: Repository inventory ==")
        repository_inventory_mod.write_repository_inventory(REPO_ROOT, output_dir)
        repo_inventory = repository_inventory_mod.build_repository_inventory(REPO_ROOT)

        print("== Step 2: Cloud Storage inventory (read-only list) ==")
        try:
            raw_objects = cloud_inventory_mod.list_bucket_objects_json(args.bucket)
        except RuntimeError as exc:
            _fail("authentication_or_listing", str(exc))
        cloud_inv = cloud_inventory_mod.build_cloud_inventory(args.bucket, raw_objects)
        cloud_inventory_mod.write_cloud_inventory(cloud_inv, output_dir)

        print("== Step 3: Download & profile known master datasets (read-only) ==")
        downloaded = download_known_master_files(args.bucket, local_data_dir)
        profiles, metadata_comparison = load_profiles(downloaded)
        dataset_profile_mod.write_dataset_profile_reports(profiles, metadata_comparison, output_dir)

        print("== Step 4: Historical coverage matrix ==")
        matrix = coverage_matrix_mod.build_coverage_matrix(profiles, repo_inventory, cloud_inv)
        coverage_matrix_mod.write_coverage_matrix(matrix, output_dir)

        print("== Step 5: Readiness decisions ==")
        readiness = readiness_mod.build_readiness_decisions(matrix, profiles)
        readiness_mod.write_readiness_decisions(readiness, output_dir)

        print("== Step 6: Job summary ==")
        summary = job_summary_mod.render_job_summary(cloud_inv, profiles, matrix, readiness)
        summary_env_path = REPO_ROOT / "artifacts" / "audits" / "job_summary.md"
        summary_env_path.write_text(summary, encoding="utf-8")

        import os
        step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary_path:
            with open(step_summary_path, "a", encoding="utf-8") as fh:
                fh.write(summary)
        else:
            print(summary)

    except AuditError as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1

    print("Audit completed successfully. Reports written to:", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
