#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from atlas.schedule.mlb_schedule_reference import normalize_schedule
from atlas.schedule.schedule_fixture_certification import certify_schedule_file
from atlas.rosters.roster_source_build import build_roster_source_bundle, write_roster_source_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.season != 2024:
        raise ValueError("this certified builder currently supports season 2024 only")
    report = certify_schedule_file(args.schedule)
    if report["verdict"] != "certified":
        raise ValueError("schedule certification failed: " + "; ".join(report["errors"]))
    raw_bytes = args.schedule.read_bytes()
    payload = json.loads(raw_bytes)
    rows = normalize_schedule(payload)
    bundle = build_roster_source_bundle(rows, season=args.season)
    bundle["schedule_source_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    bundle["schedule_unique_games"] = len(rows)
    bundle["schedule_certification_verdict"] = report["verdict"]
    manifest = write_roster_source_bundle(bundle, args.output, args.season)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
