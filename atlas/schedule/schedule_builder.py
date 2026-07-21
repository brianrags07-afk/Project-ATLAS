"""Production orchestration for deterministic historical schedule builds."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from atlas.schedule.mlb_schedule_reference import (
    CANONICAL_FIELDS,
    DETAILED_STATE_CATEGORY,
    fetch_schedule_raw,
    normalize_schedule,
)

REQUIRED_COLUMNS = tuple(CANONICAL_FIELDS) + (
    "content_hash",
    "source",
    "source_url",
    "retrieved_at_utc",
)
ARTIFACT_NAMES = (
    "canonical_schedule.parquet",
    "schedule_validation.json",
    "schedule_manifest.json",
)


class ScheduleBuildError(RuntimeError):
    """Raised when a schedule build fails validation."""


@dataclass(frozen=True)
class BuildSummary:
    seasons_built: tuple[int, ...]
    games_processed: int
    duplicate_count: int
    validation_status: str
    artifact_locations: dict[str, str]
    elapsed_build_time_seconds: float
    timestamp: str


def _season_dates(season: int) -> tuple[str, str]:
    return f"{season}-03-01", f"{season}-11-30"


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ScheduleBuildError(
            "Writing schedule artifacts requires pandas and a parquet engine"
        ) from exc
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    frame.to_parquet(path, index=False)


def validate_schedule(
    rows: Sequence[Mapping[str, Any]],
    *,
    duplicate_count: int = 0,
) -> dict[str, Any]:
    """Validate normalized rows before any artifact is published."""
    columns = tuple(rows[0].keys()) if rows else REQUIRED_COLUMNS
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
    unexpected_columns = sorted(set(columns) - set(REQUIRED_COLUMNS))
    game_pks = [row.get("game_pk") for row in rows]
    duplicate_ids = sorted(
        {game_pk for game_pk in game_pks if game_pk is not None and game_pks.count(game_pk) > 1}
    )
    invalid_statuses = sorted(
        {
            row.get("detailed_state")
            for row in rows
            if row.get("detailed_state") not in DETAILED_STATE_CATEGORY
        },
        key=str,
    )
    missing_game_pks = sum(game_pk is None for game_pk in game_pks)
    errors: list[str] = []
    if duplicate_count or duplicate_ids:
        errors.append("duplicate gamePk values detected")
    if missing_columns:
        errors.append("required columns are missing")
    if unexpected_columns:
        errors.append("schema changed unexpectedly")
    if invalid_statuses:
        errors.append("invalid schedule statuses detected")
    if missing_game_pks:
        errors.append("game_pk is missing")
    if any(not isinstance(row.get("content_hash"), str) for row in rows):
        errors.append("normalization failed")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "duplicate_count": duplicate_count + len(duplicate_ids),
        "duplicate_game_pks": duplicate_ids,
        "invalid_statuses": invalid_statuses,
        "games_processed": len(rows),
        "schema_columns": list(columns),
        "required_columns": list(REQUIRED_COLUMNS),
    }


def _raw_games(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        game
        for date_entry in payload.get("dates", []) or []
        for game in (date_entry.get("games", []) or [])
    ]


def build_historical_schedule(
    seasons: Iterable[int],
    output_dir: str | os.PathLike[str],
    *,
    fetcher: Callable[..., Mapping[str, Any]] = fetch_schedule_raw,
    timestamp: datetime | None = None,
) -> BuildSummary:
    """Fetch, normalize, validate, and write historical schedule artifacts."""
    started = time.monotonic()
    build_timestamp = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    retrieved_at_utc = build_timestamp.isoformat()
    season_list = tuple(sorted({int(season) for season in seasons}))
    if not season_list:
        raise ScheduleBuildError("At least one season is required")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payloads: list[Mapping[str, Any]] = []
    raw_ids: list[Any] = []
    for season in season_list:
        start_date, end_date = _season_dates(season)
        payload = fetcher(start_date, end_date)
        payloads.append(payload)
        raw_ids.extend(game.get("gamePk") for game in _raw_games(payload))
    raw_duplicate_count = len(raw_ids) - len(set(raw_ids))
    rows = normalize_schedule(
        payloads,
        retrieved_at_utc=retrieved_at_utc,
    )
    rows.sort(key=lambda row: (
        row.get("game_date_utc") or "",
        row.get("official_date") or "",
        row.get("game_pk") if row.get("game_pk") is not None else -1,
    ))
    validation = validate_schedule(rows, duplicate_count=raw_duplicate_count)
    validation["validated_at_utc"] = retrieved_at_utc
    validation_path = output_path / "schedule_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if validation["status"] != "passed":
        raise ScheduleBuildError("; ".join(validation["errors"]))

    canonical_path = output_path / "canonical_schedule.parquet"
    _write_parquet(rows, canonical_path)
    artifact_locations = {
        name: str(output_path / name) for name in ARTIFACT_NAMES
    }
    manifest = {
        "builder": "atlas.schedule.schedule_builder",
        "builder_version": "1",
        "seasons": list(season_list),
        "games_processed": len(rows),
        "canonical_fields": list(REQUIRED_COLUMNS),
        "artifacts": artifact_locations,
        "artifact_hashes": {
            "canonical_schedule.parquet": hashlib.sha256(
                canonical_path.read_bytes()
            ).hexdigest()
        },
        "schedule_history_implemented": False,
        "source": "mlb_stats_api_schedule",
        "git_revision": _git_revision(),
        "manifest_content_hash": None,
    }
    manifest["manifest_content_hash"] = _stable_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    )
    (output_path / "schedule_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return BuildSummary(
        seasons_built=season_list,
        games_processed=len(rows),
        duplicate_count=validation["duplicate_count"],
        validation_status=validation["status"],
        artifact_locations=artifact_locations,
        elapsed_build_time_seconds=time.monotonic() - started,
        timestamp=retrieved_at_utc,
    )
