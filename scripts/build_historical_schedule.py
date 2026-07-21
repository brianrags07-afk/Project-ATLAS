#!/usr/bin/env python3
"""CLI entry point for the ATLAS historical schedule builder."""

from __future__ import annotations

import argparse
import json

from atlas.schedule.schedule_builder import build_historical_schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        required=True,
        help="Season to build; repeat this option for multiple seasons.",
    )
    parser.add_argument("--output", required=True, help="Artifact output directory.")
    args = parser.parse_args()
    summary = build_historical_schedule(args.season, args.output)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
