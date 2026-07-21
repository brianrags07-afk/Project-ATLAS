#!/usr/bin/env python3
"""Certify historical datasets against a published MLB schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from atlas.audit.historical_dataset_certification import (
    certify_historical_datasets,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--pitch", required=True)
    parser.add_argument("--team-state", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.schedule).open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    report = certify_historical_datasets(
        schedule,
        pd.read_parquet(args.master),
        pd.read_parquet(args.pitch),
        pd.read_parquet(args.team_state),
        season=args.season,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": report["verdict"], "errors": report["errors"]}))
    return 0 if report["verdict"].startswith("certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
