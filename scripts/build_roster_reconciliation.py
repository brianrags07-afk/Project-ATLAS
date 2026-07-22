#!/usr/bin/env python3
"""Build immutable, prospective reconciliation evidence for 2024 rosters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.rosters.player_observations import (
    build_postgame_player_observations,
    certify_postgame_player_observations,
)
from atlas.rosters.roster_reconciliation import reconcile_quarantined_transactions
from atlas.schedule.mlb_schedule_reference import normalize_schedule


EVENT_COLUMNS = [
    "game_pk",
    "atlas_season",
    "game_type",
    "inning_topbot",
    "batter",
    "pitcher",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path, *, season: int, label: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("season") or 0) != int(season):
        raise ValueError(f"{label} is not for season {season}")
    if manifest.get("game_type") != "R":
        raise ValueError(f"{label} is not regular-season isolated")
    return manifest


def _verify_manifest_artifact(
    path: Path,
    manifest: dict[str, Any],
    artifact_name: str,
    label: str,
) -> str:
    expected = (
        manifest.get("artifacts", {}).get(artifact_name, {}).get("sha256")
    )
    if not expected:
        raise ValueError(f"{label} does not declare {artifact_name}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} checksum mismatch: {artifact_name}")
    return observed


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--teams", required=True, type=Path)
    parser.add_argument("--roster-source-manifest", required=True, type=Path)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--event-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-events-sha256")
    parser.add_argument("--expected-schedule-sha256")
    args = parser.parse_args()

    if args.season != 2024:
        raise ValueError("this certified reconciliation build currently supports 2024 only")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_manifest = _load_manifest(
        args.roster_source_manifest,
        season=args.season,
        label="roster source manifest",
    )
    event_manifest = _load_manifest(
        args.event_manifest,
        season=args.season,
        label="roster event manifest",
    )
    team_sha = _verify_manifest_artifact(
        args.teams, source_manifest, "teams.parquet", "roster source manifest"
    )
    quarantine_sha = _verify_manifest_artifact(
        args.quarantine,
        event_manifest,
        "roster_event_quarantine.parquet",
        "roster event manifest",
    )

    events_sha = sha256(args.events)
    schedule_sha = sha256(args.schedule)
    if args.expected_events_sha256 and events_sha != args.expected_events_sha256:
        raise ValueError("certified pitch-event source checksum mismatch")
    if args.expected_schedule_sha256 and schedule_sha != args.expected_schedule_sha256:
        raise ValueError("certified schedule source checksum mismatch")
    if source_manifest.get("schedule_source_sha256") != schedule_sha:
        raise ValueError("roster source and reconciliation schedule hashes do not match")
    if event_manifest.get("source_schedule_sha256") != schedule_sha:
        raise ValueError("roster event and reconciliation schedule hashes do not match")

    raw_schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    schedule = pd.DataFrame(normalize_schedule(raw_schedule))
    events = pd.read_parquet(args.events, columns=EVENT_COLUMNS)
    teams = pd.read_parquet(args.teams)
    quarantine = pd.read_parquet(args.quarantine)

    observations = build_postgame_player_observations(
        events,
        schedule,
        teams,
        season=args.season,
    )
    observation_certification = certify_postgame_player_observations(
        observations, season=args.season
    )
    if observation_certification["verdict"] != "certified":
        raise ValueError(
            "observation certification failed: "
            + "; ".join(observation_certification["errors"])
        )

    reconciliation = reconcile_quarantined_transactions(
        quarantine, observations
    )
    if len(reconciliation) != len(quarantine):
        raise AssertionError("reconciliation did not preserve quarantine row count")
    if reconciliation["retroactive_backfill_allowed"].fillna(True).astype(bool).any():
        raise AssertionError("reconciliation unexpectedly permits retroactive backfill")

    observation_path = args.output / "player_observations.parquet"
    reconciliation_path = args.output / "roster_reconciliation.parquet"
    observations.to_parquet(observation_path, index=False)
    reconciliation.to_parquet(reconciliation_path, index=False)

    status_counts = _value_counts(reconciliation["reconciliation_status"])
    known_identity = quarantine["player_id"].notna()
    observed_statuses = {
        "same_scoped_team_observed",
        "different_team_observed",
    }
    prospectively_observed_rows = int(
        reconciliation["reconciliation_status"].isin(observed_statuses).sum()
    )
    summary = {
        "season": args.season,
        "game_type": "R",
        "purpose": "prospective evidence for quarantined roster facts",
        "observation_certification": observation_certification,
        "quarantine_rows_preserved": int(len(quarantine)),
        "quarantine_known_identity_rows": int(known_identity.sum()),
        "reconciliation_rows": int(len(reconciliation)),
        "reconciliation_status_counts": status_counts,
        "prospectively_observed_rows": prospectively_observed_rows,
        "prospective_observation_rate_among_quarantine": (
            prospectively_observed_rows / len(quarantine) if len(quarantine) else 0.0
        ),
        "roster_event_ledger_modified": False,
        "same_game_pregame_use_allowed": False,
        "retroactive_backfill_allowed": False,
        "prediction_created": False,
    }
    summary_path = args.output / "reconciliation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    artifacts = {}
    for path in (observation_path, reconciliation_path, summary_path):
        artifact = {"sha256": sha256(path)}
        if path.suffix == ".parquet":
            artifact["rows"] = int(
                len(observations if path == observation_path else reconciliation)
            )
        artifacts[path.name] = artifact

    manifest = {
        "season": args.season,
        "game_type": "R",
        "build_class": "immutable_audit_evidence",
        "source_hashes": {
            "pitch_events": events_sha,
            "schedule": schedule_sha,
            "teams": team_sha,
            "roster_event_quarantine": quarantine_sha,
            "roster_source_manifest": sha256(args.roster_source_manifest),
            "roster_event_manifest": sha256(args.event_manifest),
        },
        "artifacts": artifacts,
        "roster_event_ledger_modified": False,
        "retroactive_backfill_allowed": False,
        "promotion_status": "audit_only_not_roster_mutation",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
