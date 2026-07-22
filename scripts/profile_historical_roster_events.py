#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    return None if pd.isna(value) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["season"] != 2024 or manifest["certification"]["verdict"] != "certified":
        raise ValueError("roster event source is not certified for 2024")
    for name in ("roster_events.parquet", "roster_event_quarantine.parquet"):
        if digest(args.source / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {name}")
    events = pd.read_parquet(args.source / "roster_events.parquet")
    quarantine = pd.read_parquet(args.source / "roster_event_quarantine.parquet")
    event_counts = (events.groupby(["event_type", "team"], dropna=False).size()
                    .rename("row_count").reset_index())
    dimensions = [column for column in ["quarantine_source", "quarantine_reason", "type_code", "type_description"] if column in quarantine.columns]
    if quarantine.empty:
        quarantine_counts = pd.DataFrame(columns=[*dimensions, "row_count", "unique_players"])
    else:
        quarantine_counts = (quarantine.groupby(dimensions, dropna=False)
            .agg(row_count=("quarantine_reason", "size"), unique_players=("player_id", "nunique"))
            .reset_index())
    event_counts.to_csv(args.output / "event_counts.csv", index=False)
    quarantine_counts.to_csv(args.output / "quarantine_type_profile.csv", index=False)
    profile = {
        "season": 2024, "source_manifest_sha256": digest(manifest_path),
        "source_certification": manifest["certification"]["verdict"],
        "event_rows": int(len(events)), "quarantine_rows": int(len(quarantine)),
        "event_types": {str(clean(k)): int(v) for k, v in events["event_type"].value_counts(dropna=False).items()},
        "quarantine_reasons": {str(clean(k)): int(v) for k, v in quarantine["quarantine_reason"].value_counts(dropna=False).items()} if not quarantine.empty else {},
        "transaction_type_codes": {str(clean(k)): int(v) for k, v in quarantine.get("type_code", pd.Series(dtype="object")).value_counts(dropna=False).items()},
        "semantic_mapping_status": "team_scoped_status_semantics_v3",
    }
    profile_path = args.output / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True))
    output_manifest = {
        "season": 2024, "source_manifest_sha256": profile["source_manifest_sha256"],
        "artifacts": {name: {"sha256": digest(args.output / name)} for name in ("event_counts.csv", "quarantine_type_profile.csv", "profile.json")},
        "promotion_status": "audit_only_not_canonical",
    }
    (args.output / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True))
    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
