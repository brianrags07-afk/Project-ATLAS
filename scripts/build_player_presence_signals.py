#!/usr/bin/env python3
"""Build immutable 2024 pregame player-presence evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.rosters.player_presence_signals import (
    build_pregame_player_presence_signals,
    certify_pregame_player_presence_signals,
)


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


def value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--team-games", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--observation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.season != 2024:
        raise ValueError("this certified player-presence build currently supports 2024 only")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    snapshot_manifest = load_manifest(
        args.snapshot_manifest,
        season=args.season,
        label="roster snapshot manifest",
    )
    observation_manifest = load_manifest(
        args.observation_manifest,
        season=args.season,
        label="player observation manifest",
    )
    snapshot_sha = verify_artifact(
        args.snapshots,
        snapshot_manifest,
        "pregame_roster_snapshots.parquet",
        "roster snapshot manifest",
    )
    team_game_sha = verify_artifact(
        args.team_games,
        snapshot_manifest,
        "team_games.parquet",
        "roster snapshot manifest",
    )
    observation_sha = verify_artifact(
        args.observations,
        observation_manifest,
        "player_observations.parquet",
        "player observation manifest",
    )

    snapshots = pd.read_parquet(args.snapshots)
    team_games = pd.read_parquet(args.team_games)
    observations = pd.read_parquet(args.observations)
    signals = build_pregame_player_presence_signals(
        snapshots,
        observations,
        team_games,
        season=args.season,
    )
    certification = certify_pregame_player_presence_signals(
        signals, season=args.season
    )
    if certification["verdict"] != "certified":
        raise ValueError(
            "player-presence certification failed: "
            + "; ".join(certification["errors"])
        )

    signal_path = args.output / "pregame_player_presence_signals.parquet"
    signals.to_parquet(signal_path, index=False)

    by_team_game = (
        signals.groupby(["game_pk", "team"], sort=False)
        .agg(
            presence_signal_rows=("player_id", "size"),
            roster_rows=("roster_row_present", "sum"),
            active_roster_rows=("active_roster_known_true", "sum"),
            prior_team_appearance_rows=("prior_team_appearance_known", "sum"),
            observation_only_rows=(
                "roster_row_present",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            published_lineup_rows=("published_lineup_confirmed", "sum"),
        )
        .reset_index()
    )
    team_coverage = (
        by_team_game.groupby("team", sort=True)
        .agg(
            team_games=("game_pk", "nunique"),
            mean_presence_signal_rows=("presence_signal_rows", "mean"),
            mean_roster_rows=("roster_rows", "mean"),
            mean_active_roster_rows=("active_roster_rows", "mean"),
            mean_prior_team_appearance_rows=("prior_team_appearance_rows", "mean"),
            total_observation_only_rows=("observation_only_rows", "sum"),
            total_published_lineup_rows=("published_lineup_rows", "sum"),
        )
        .reset_index()
    )
    team_coverage_path = args.output / "team_presence_coverage.csv"
    team_coverage.to_csv(team_coverage_path, index=False)

    summary = {
        "season": args.season,
        "game_type": "R",
        "certification": certification,
        "presence_evidence_class_counts": value_counts(
            signals["presence_evidence_class"]
        ),
        "active_roster_known_true_rows": int(
            signals["active_roster_known_true"].sum()
        ),
        "prior_team_appearance_known_rows": int(
            signals["prior_team_appearance_known"].sum()
        ),
        "latest_observation_other_team_rows": int(
            signals["latest_observation_other_team"].sum()
        ),
        "published_lineup_confirmed_rows": int(
            signals["published_lineup_confirmed"].sum()
        ),
        "same_game_postgame_used_rows": int(
            signals["same_game_postgame_used"].sum()
        ),
        "future_games_used_rows": int(signals["future_games_used"].sum()),
        "roster_ledger_mutated_rows": int(signals["roster_ledger_mutated"].sum()),
        "prediction_created": False,
    }
    summary_path = args.output / "player_presence_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    artifacts = {}
    for path, rows in (
        (signal_path, len(signals)),
        (team_coverage_path, len(team_coverage)),
        (summary_path, None),
    ):
        entry = {"sha256": sha256(path)}
        if rows is not None:
            entry["rows"] = int(rows)
        artifacts[path.name] = entry

    manifest = {
        "season": args.season,
        "game_type": "R",
        "build_class": "immutable_pregame_player_presence_evidence",
        "certification": certification,
        "source_hashes": {
            "pregame_roster_snapshots": snapshot_sha,
            "team_games": team_game_sha,
            "player_observations": observation_sha,
            "roster_snapshot_manifest": sha256(args.snapshot_manifest),
            "player_observation_manifest": sha256(args.observation_manifest),
        },
        "artifacts": artifacts,
        "published_lineup_inferred": False,
        "future_games_used": False,
        "same_game_postgame_data_used": False,
        "roster_ledger_mutated": False,
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
