#!/usr/bin/env python3
"""Build immutable 2024 roster state at each completed game's first pitch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.rosters.roster_snapshot_build import (
    build_regular_team_games,
    certify_pregame_roster_snapshots,
)
from atlas.rosters.roster_timeline import (
    build_pregame_roster_snapshots,
    certify_roster_events,
)
from atlas.schedule.mlb_schedule_reference import normalize_schedule


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path, *, season: int, label: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("season") or 0) != int(season):
        raise ValueError(f"{label} is not for season {season}")
    if manifest.get("game_type") != "R":
        raise ValueError(f"{label} is not regular-season isolated")
    return manifest


def verify_artifact(
    path: Path,
    manifest: dict[str, Any],
    name: str,
    label: str,
) -> str:
    expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
    if not expected:
        raise ValueError(f"{label} does not declare {name}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} checksum mismatch: {name}")
    return observed


def counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--teams", required=True, type=Path)
    parser.add_argument("--roster-source-manifest", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--event-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-schedule-sha256")
    parser.add_argument("--expected-completed-games", type=int)
    args = parser.parse_args()

    if args.season != 2024:
        raise ValueError("this certified roster snapshot build currently supports 2024 only")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_manifest = load_manifest(
        args.roster_source_manifest,
        season=args.season,
        label="roster source manifest",
    )
    event_manifest = load_manifest(
        args.event_manifest,
        season=args.season,
        label="roster event manifest",
    )
    team_sha = verify_artifact(
        args.teams, source_manifest, "teams.parquet", "roster source manifest"
    )
    event_sha = verify_artifact(
        args.events, event_manifest, "roster_events.parquet", "roster event manifest"
    )

    schedule_sha = sha256(args.schedule)
    if args.expected_schedule_sha256 and schedule_sha != args.expected_schedule_sha256:
        raise ValueError("certified schedule source checksum mismatch")
    if source_manifest.get("schedule_source_sha256") != schedule_sha:
        raise ValueError("roster source and snapshot schedule hashes do not match")
    if event_manifest.get("source_schedule_sha256") != schedule_sha:
        raise ValueError("roster event and snapshot schedule hashes do not match")

    raw_schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    schedule = pd.DataFrame(normalize_schedule(raw_schedule))
    teams = pd.read_parquet(args.teams)
    events = pd.read_parquet(args.events)
    event_certification = certify_roster_events(events)
    if event_certification["verdict"] != "certified":
        raise ValueError(
            "roster event certification failed: "
            + "; ".join(event_certification["errors"])
        )

    team_games = build_regular_team_games(
        schedule, teams, season=args.season
    )
    completed_games = int(team_games["game_pk"].nunique())
    if (
        args.expected_completed_games is not None
        and completed_games != args.expected_completed_games
    ):
        raise ValueError(
            "completed regular-season game count mismatch: "
            f"{completed_games} != {args.expected_completed_games}"
        )

    snapshots = build_pregame_roster_snapshots(
        events,
        team_games[["game_pk", "game_start_at", "season", "team"]],
    )
    snapshot_certification = certify_pregame_roster_snapshots(
        snapshots, team_games, season=args.season
    )
    if snapshot_certification["verdict"] != "certified":
        raise ValueError(
            "pregame roster snapshot certification failed: "
            + "; ".join(snapshot_certification["errors"])
        )

    team_games_path = args.output / "team_games.parquet"
    snapshots_path = args.output / "pregame_roster_snapshots.parquet"
    team_games.to_parquet(team_games_path, index=False)
    snapshots.to_parquet(snapshots_path, index=False)

    summary = {
        "season": args.season,
        "game_type": "R",
        "event_certification": event_certification,
        "snapshot_certification": snapshot_certification,
        "completed_games": completed_games,
        "team_games": int(len(team_games)),
        "snapshot_rows": int(len(snapshots)),
        "active_roster_counts": counts(snapshots["active_roster"]),
        "availability_counts": counts(snapshots["available"]),
        "roster_status_counts": counts(snapshots["roster_status"]),
        "last_event_type_counts": counts(snapshots["last_event_type"]),
        "pregame_safe_rows": int(snapshots["pregame_safe"].sum()),
        "post_first_pitch_event_rows": 0,
        "post_first_pitch_knowledge_rows": 0,
        "prediction_created": False,
    }
    summary_path = args.output / "roster_snapshot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    artifacts = {}
    for path, rows in (
        (team_games_path, len(team_games)),
        (snapshots_path, len(snapshots)),
        (summary_path, None),
    ):
        entry = {"sha256": sha256(path)}
        if rows is not None:
            entry["rows"] = int(rows)
        artifacts[path.name] = entry

    manifest = {
        "season": args.season,
        "game_type": "R",
        "build_class": "immutable_pregame_roster_snapshot",
        "certification": snapshot_certification,
        "source_hashes": {
            "schedule": schedule_sha,
            "teams": team_sha,
            "roster_events": event_sha,
            "roster_source_manifest": sha256(args.roster_source_manifest),
            "roster_event_manifest": sha256(args.event_manifest),
        },
        "artifacts": artifacts,
        "future_games_used": False,
        "same_game_postgame_data_used": False,
        "prediction_created": False,
        "promotion_status": "build_only_not_canonical",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
