"""Season-isolated builder for official MLB roster source facts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from atlas.rosters.mlb_roster_source import (
    fetch_roster, fetch_team_directory, fetch_transactions,
    normalize_roster, normalize_teams, normalize_transactions,
)


def schedule_team_windows(rows: Iterable[Mapping[str, Any]], season: int) -> pd.DataFrame:
    records = []
    for row in rows:
        if int(row.get("season") or 0) != int(season) or row.get("game_type_code") != "R":
            continue
        game_date = row.get("official_date")
        if not game_date:
            raise ValueError("regular-season schedule row is missing official_date")
        for side in ("home", "away"):
            records.append({"team_id": row[f"{side}_team_id"], "game_date": game_date})
    frame = pd.DataFrame(records)
    if frame.empty or frame["team_id"].isna().any():
        raise ValueError("schedule contains no complete regular-season team windows")
    return (frame.groupby("team_id", as_index=False)["game_date"]
            .agg(first_game_date="min", last_game_date="max")
            .sort_values("team_id").reset_index(drop=True))


def build_roster_source_bundle(
    schedule_rows: Iterable[Mapping[str, Any]], *, season: int,
    retrieved_at_utc: str | None = None,
    team_fetch: Callable[..., Mapping[str, Any]] = fetch_team_directory,
    roster_fetch: Callable[..., Mapping[str, Any]] = fetch_roster,
    transaction_fetch: Callable[..., Mapping[str, Any]] = fetch_transactions,
) -> dict[str, Any]:
    retrieved = retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    windows = schedule_team_windows(schedule_rows, season)
    team_payload = team_fetch(season)
    teams = normalize_teams(team_payload, season)
    scheduled_ids = set(windows["team_id"].astype(int))
    directory_ids = set(teams["team_id"].astype(int))
    if scheduled_ids != directory_ids:
        raise ValueError(f"schedule/team directory mismatch: missing={sorted(scheduled_ids-directory_ids)}, unexpected={sorted(directory_ids-scheduled_ids)}")

    roster_frames, transaction_frames, raw = [], [], {"teams": team_payload, "clubs": {}}
    for window in windows.to_dict("records"):
        team_id = int(window["team_id"])
        club_raw = {"rosters": {}, "transactions": None}
        for roster_type in ("active", "40Man"):
            payload = roster_fetch(team_id, window["first_game_date"], roster_type)
            club_raw["rosters"][roster_type] = payload
            roster_frames.append(normalize_roster(payload, season=season, team_id=team_id,
                as_of_date=window["first_game_date"], roster_type=roster_type,
                retrieved_at_utc=retrieved))
        payload = transaction_fetch(team_id, window["first_game_date"], window["last_game_date"])
        club_raw["transactions"] = payload
        transaction_frames.append(normalize_transactions(payload, season=season,
            requested_team_id=team_id, retrieved_at_utc=retrieved))
        raw["clubs"][str(team_id)] = club_raw

    rosters = pd.concat(roster_frames, ignore_index=True)
    transactions = pd.concat(transaction_frames, ignore_index=True)
    return {"teams": teams, "team_windows": windows, "rosters": rosters,
            "transactions": transactions, "raw_payloads": raw, "retrieved_at_utc": retrieved}


def write_roster_source_bundle(bundle: Mapping[str, Any], output_dir: str | Path, season: int) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ("teams", "team_windows", "rosters", "transactions"):
        path = output / f"{name}.parquet"
        bundle[name].to_parquet(path, index=False)
        artifacts[path.name] = {"rows": len(bundle[name]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    raw_path = output / "raw_payloads.json"
    raw_path.write_text(json.dumps(bundle["raw_payloads"], sort_keys=True, indent=2))
    artifacts[raw_path.name] = {"sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    manifest = {"season": int(season), "game_type": "R", "retrieved_at_utc": bundle["retrieved_at_utc"],
                "team_count": len(bundle["team_windows"]), "artifacts": artifacts,
                "schedule_source_sha256": bundle["schedule_source_sha256"],
                "schedule_unique_games": bundle["schedule_unique_games"],
                "schedule_certification_verdict": bundle["schedule_certification_verdict"],
                "promotion_status": "build_only_not_canonical"}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest
