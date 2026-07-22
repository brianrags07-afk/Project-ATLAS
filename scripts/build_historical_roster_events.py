#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from atlas.rosters.roster_event_conversion import directional_transaction_events, opening_roster_events
from atlas.rosters.roster_timeline import certify_roster_events


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--season", required=True, type=int)
    args = parser.parse_args()
    if args.season != 2024:
        raise ValueError("this certified event build currently supports 2024 only")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((args.source / "manifest.json").read_text())
    if source_manifest["season"] != args.season or source_manifest["schedule_certification_verdict"] != "certified":
        raise ValueError("source roster bundle is not certified for requested season")
    for name in ("teams.parquet", "rosters.parquet", "transactions.parquet"):
        observed = sha256(args.source / name)
        expected = source_manifest["artifacts"][name]["sha256"]
        if observed != expected:
            raise ValueError(f"source artifact checksum mismatch: {name}")

    teams = pd.read_parquet(args.source / "teams.parquet")
    rosters = pd.read_parquet(args.source / "rosters.parquet")
    transactions = pd.read_parquet(args.source / "transactions.parquet")
    opening, opening_quarantine = opening_roster_events(rosters, teams)
    transfers, transaction_quarantine = directional_transaction_events(transactions, teams)
    events = pd.concat([opening, transfers], ignore_index=True).sort_values(
        ["knowledge_available_at", "team", "player_id", "event_id"], kind="stable"
    ).reset_index(drop=True)
    quarantine = pd.concat([
        opening_quarantine.assign(quarantine_source="opening_roster"),
        transaction_quarantine.assign(quarantine_source="transaction"),
    ], ignore_index=True, sort=False)
    certification = certify_roster_events(events)
    if certification["verdict"] != "certified":
        raise ValueError("event certification failed: " + "; ".join(certification["errors"]))

    event_path = args.output / "roster_events.parquet"
    quarantine_path = args.output / "roster_event_quarantine.parquet"
    events.to_parquet(event_path, index=False)
    quarantine.to_parquet(quarantine_path, index=False)
    manifest = {
        "season": args.season, "game_type": "R", "certification": certification,
        "source_manifest_sha256": sha256(args.source / "manifest.json"),
        "source_schedule_sha256": source_manifest["schedule_source_sha256"],
        "event_rows": len(events), "quarantine_rows": len(quarantine),
        "artifacts": {
            event_path.name: {"rows": len(events), "sha256": sha256(event_path)},
            quarantine_path.name: {"rows": len(quarantine), "sha256": sha256(quarantine_path)},
        },
        "promotion_status": "build_only_not_canonical",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
