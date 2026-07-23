#!/usr/bin/env python3
"""Build an immutable official MLB historical pregame lineup source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.lineups.mlb_pregame_lineup_source import (
    GAME_SNAPSHOT_COLUMNS,
    LINEUP_COLUMNS,
    STARTER_COLUMNS,
    certify_timecoded_pregame_bundle,
    fetch_timecoded_game_feed,
    format_mlb_timecode,
    normalize_timecoded_pregame_feed,
    partition_timecoded_pregame_bundle,
    prepare_completed_regular_games,
)
from atlas.schedule.mlb_schedule_reference import normalize_schedule
from atlas.schedule.schedule_fixture_certification import certify_schedule_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _chunks(records: list[dict[str, Any]], size: int):
    for offset in range(0, len(records), size):
        yield records[offset : offset + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--expected-completed-games", required=True, type=int)
    parser.add_argument("--cutoff-minutes", default=15, type=int)
    parser.add_argument("--max-workers", default=6, type=int)
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        raise ValueError("season is outside the supported MLB year range")
    if args.cutoff_minutes <= 0:
        raise ValueError("cutoff-minutes must be positive")
    if args.max_workers < 1 or args.max_workers > 12:
        raise ValueError("max-workers must be from 1 through 12")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    schedule_sha = sha256(args.schedule)
    if schedule_sha != args.expected_schedule_sha256:
        raise ValueError("certified schedule source checksum mismatch")
    schedule_certification = certify_schedule_file(args.schedule)
    if schedule_certification["verdict"] != "certified":
        raise ValueError(
            "schedule certification failed: "
            + "; ".join(schedule_certification["errors"])
        )
    raw_schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    normalized_schedule = normalize_schedule(raw_schedule)
    games = prepare_completed_regular_games(
        normalized_schedule, season=args.season
    )
    if games["game_pk"].nunique() != args.expected_completed_games:
        raise ValueError(
            "completed regular-season game count mismatch: "
            f"{games['game_pk'].nunique()} != {args.expected_completed_games}"
        )

    archive_retrieved_at = datetime.now(timezone.utc).isoformat()
    game_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    starter_rows: list[dict[str, Any]] = []
    raw_index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_path = args.output / "raw_timecoded_pregame_feeds.jsonl.gz"
    game_records = games.to_dict("records")
    batch_size = args.max_workers * 4

    with raw_path.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                processed = 0
                for batch in _chunks(game_records, batch_size):
                    futures = []
                    for game in batch:
                        cutoff = pd.to_datetime(
                            game["game_start_at"], utc=True
                        ) - pd.Timedelta(minutes=args.cutoff_minutes)
                        timecode = format_mlb_timecode(cutoff)
                        futures.append(
                            (
                                game,
                                timecode,
                                executor.submit(
                                    fetch_timecoded_game_feed,
                                    int(game["game_pk"]),
                                    timecode,
                                ),
                            )
                        )
                    for game, timecode, future in futures:
                        game_pk = int(game["game_pk"])
                        try:
                            payload = future.result()
                            payload_bytes = _canonical_payload_bytes(payload)
                            envelope = _canonical_payload_bytes(
                                {
                                    "game_pk": game_pk,
                                    "requested_timecode": timecode,
                                    "payload": payload,
                                }
                            )
                            compressed.write(envelope + b"\n")
                            snapshot, lineups, starters = (
                                normalize_timecoded_pregame_feed(
                                    payload,
                                    game,
                                    cutoff_minutes=args.cutoff_minutes,
                                    archive_retrieved_at=archive_retrieved_at,
                                )
                            )
                            game_rows.append(snapshot)
                            lineup_rows.extend(lineups)
                            starter_rows.extend(starters)
                            raw_index_rows.append(
                                {
                                    "game_pk": game_pk,
                                    "requested_timecode": timecode,
                                    "source_snapshot_at": snapshot[
                                        "source_snapshot_at"
                                    ],
                                    "canonical_payload_bytes": len(payload_bytes),
                                    "canonical_payload_sha256": hashlib.sha256(
                                        payload_bytes
                                    ).hexdigest(),
                                }
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "game_pk": game_pk,
                                    "requested_timecode": timecode,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                        processed += 1
                        if processed % 100 == 0 or processed == len(game_records):
                            print(
                                f"Processed {processed}/{len(game_records)} games; "
                                f"failures={len(failures)}",
                                flush=True,
                            )

    failure_path = args.output / "fetch_failures.json"
    _write_json(failure_path, failures)
    if failures:
        raise RuntimeError(
            f"MLB timecoded feed retrieval failed for {len(failures)} games; "
            f"see {failure_path}"
        )

    all_game_snapshots = pd.DataFrame(
        game_rows, columns=GAME_SNAPSHOT_COLUMNS
    ).sort_values(["game_start_at", "game_pk"], kind="stable").reset_index(
        drop=True
    )
    lineups = pd.DataFrame(lineup_rows, columns=LINEUP_COLUMNS).sort_values(
        ["game_start_at", "game_pk", "home_away", "batting_order"],
        kind="stable",
    ).reset_index(drop=True)
    starters = pd.DataFrame(starter_rows, columns=STARTER_COLUMNS).sort_values(
        ["game_start_at", "game_pk", "home_away"], kind="stable"
    ).reset_index(drop=True)
    raw_index = pd.DataFrame(raw_index_rows).sort_values(
        ["game_pk"], kind="stable"
    ).reset_index(drop=True)
    for frame in (all_game_snapshots, lineups, starters, raw_index):
        for column in (
            "game_pk",
            "season",
            "team_id",
            "opponent_team_id",
            "batting_order",
            "player_id",
            "pitcher_id",
            "home_probable_pitcher_id",
            "away_probable_pitcher_id",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                ).astype("Int64")

    (
        game_snapshots,
        lineups,
        starters,
        quarantined_game_snapshots,
    ) = partition_timecoded_pregame_bundle(
        all_game_snapshots,
        lineups,
        starters,
    )
    certification = certify_timecoded_pregame_bundle(
        all_game_snapshots,
        lineups,
        starters,
        games,
        season=args.season,
    )
    if certification["verdict"] not in {
        "certified",
        "certified_with_documented_gaps",
    }:
        raise ValueError(
            "historical pregame source certification failed: "
            + "; ".join(certification["errors"])
        )

    paths = {
        "pregame_game_snapshots.parquet": game_snapshots,
        "pregame_lineup_snapshots.parquet": lineups,
        "pregame_probable_starter_snapshots.parquet": starters,
        "quarantined_game_snapshots.parquet": quarantined_game_snapshots,
        "raw_payload_index.parquet": raw_index,
    }
    for name, frame in paths.items():
        frame.to_parquet(args.output / name, index=False)

    summary = {
        "season": args.season,
        "game_type": "R",
        "cutoff_minutes_before_start": args.cutoff_minutes,
        "capture_mode": "official_historical_replay",
        "archive_retrieved_at": archive_retrieved_at,
        "schedule_certification": schedule_certification,
        "source_certification": certification,
        "published_lineup_team_coverage_pct": (
            certification["complete_team_lineups"]
            / certification["team_games"]
            * 100
        ),
        "probable_starter_coverage_pct": (
            certification["probable_starter_rows"]
            / certification["team_games"]
            * 100
        ),
        "actual_live_capture": False,
        "historical_replay": True,
        "outcome_fields_extracted": 0,
        "quarantined_games_are_model_readable": False,
        "prediction_created": False,
    }
    summary_path = args.output / "pregame_source_summary.json"
    _write_json(summary_path, summary)

    artifacts: dict[str, dict[str, Any]] = {}
    row_counts = {
        name: int(len(frame)) for name, frame in paths.items()
    }
    row_counts["fetch_failures.json"] = 0
    for path in sorted(args.output.iterdir()):
        entry: dict[str, Any] = {"sha256": sha256(path)}
        if path.name in row_counts:
            entry["rows"] = row_counts[path.name]
        artifacts[path.name] = entry
    manifest = {
        "season": args.season,
        "game_type": "R",
        "build_class": "immutable_official_historical_pregame_lineup_source",
        "capture_mode": "official_historical_replay",
        "cutoff_minutes_before_start": args.cutoff_minutes,
        "archive_retrieved_at": archive_retrieved_at,
        "source_snapshot_timestamp_required": True,
        "actual_live_capture": False,
        "historical_replay": True,
        "normalized_outcome_fields": 0,
        "raw_payloads_model_readable": False,
        "quarantined_snapshots_model_readable": False,
        "normalized_outputs_contain_only_pregame_safe_rows": True,
        "same_game_postgame_data_used": False,
        "future_games_used": False,
        "prediction_created": False,
        "schedule_source_sha256": schedule_sha,
        "schedule_unique_games": len(normalized_schedule),
        "completed_regular_games": int(len(games)),
        "certification": certification,
        "artifacts": artifacts,
        "promotion_status": "build_only_not_canonical",
    }
    manifest_path = args.output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
