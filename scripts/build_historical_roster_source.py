#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlas.schedule.mlb_schedule_reference import normalize_schedule
from atlas.rosters.roster_source_build import build_roster_source_bundle, write_roster_source_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.schedule.read_text())
    rows = normalize_schedule(payload)
    bundle = build_roster_source_bundle(rows, season=args.season)
    manifest = write_roster_source_bundle(bundle, args.output, args.season)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
