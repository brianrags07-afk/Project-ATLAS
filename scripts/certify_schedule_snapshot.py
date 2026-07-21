#!/usr/bin/env python3
"""Certify an immutable saved MLB schedule snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlas.schedule.schedule_fixture_certification import certify_schedule_file


def _markdown(report: dict) -> str:
    lines = [
        "# ATLAS 2024 Schedule Snapshot Certification",
        "",
        f"- Verdict: **{report['verdict']}**",
        f"- Source SHA-256: `{report['source_sha256']}`",
        f"- Source bytes: {report['source_size_bytes']}",
        "",
        "## Metrics",
        "",
        "| Metric | Observed | Expected |",
        "|---|---:|---:|",
    ]
    for key, observed in report["metrics"].items():
        lines.append(
            f"| {key} | {observed} | {report['expected_metrics'].get(key, '')} |"
        )
    lines.extend(["", "## Special games", ""])
    for game_pk, values in report["special_games"].items():
        lines.append(f"- `{game_pk}`: {json.dumps(values, sort_keys=True)}")
    if report["errors"]:
        lines.extend(["", "## Certification errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = certify_schedule_file(args.schedule_json)
    (output / "schedule_snapshot_certification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "schedule_snapshot_certification.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": report["verdict"], "errors": report["errors"]}))
    return 0 if report["verdict"] == "certified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
