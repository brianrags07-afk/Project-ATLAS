"""Regular-season schedule build with lossless schedule history."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from atlas.schedule.mlb_schedule_reference import fetch_schedule_raw, normalize_schedule
from atlas.schedule.schedule_builder import (
    BuildSummary,
    ScheduleBuildError,
    _git_revision,
    _stable_json_hash,
    _write_parquet,
    validate_schedule,
)
from atlas.schedule.schedule_history import (
    HISTORY_EXTRA_FIELDS,
    build_schedule_change_audit,
    normalize_schedule_history,
    schedule_history_metrics,
)

ARTIFACT_NAMES = (
    "canonical_schedule.parquet",
    "schedule_history.parquet",
    "rescheduled_games_audit.csv",
    "schedule_validation.json",
    "schedule_manifest.json",
)


def _season_dates(season: int) -> tuple[str, str]:
    return f"{season}-03-01", f"{season}-11-30"



def _write_history_parquet(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    columns: list[str] | None,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ScheduleBuildError(
            "Writing schedule artifacts requires pandas and a parquet engine"
        ) from exc
    pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)

def _write_audit(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else [
        "game_pk",
        "season",
        "game_type_code",
        "original_scheduled_dates",
        "rescheduled_dates",
        "was_postponed",
        "was_rescheduled",
        "was_suspended_or_resumed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_regular_only(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = validate_schedule(rows)
    counts: dict[str, int] = {}
    for row in rows:
        code = str(row.get("game_type_code"))
        counts[code] = counts.get(code, 0) + 1
    unexpected = sorted(
        row["game_pk"]
        for row in rows
        if row.get("game_type_code") != "R" and row.get("game_pk") is not None
    )
    result["required_game_type"] = "R"
    result["game_type_counts"] = dict(sorted(counts.items()))
    result["non_regular_game_pks"] = unexpected
    if unexpected:
        result["status"] = "failed"
        result["errors"].append(
            "non-regular games detected in regular-season build"
        )
    return result


def build_historical_schedule_v2(
    seasons: Iterable[int],
    output_dir: str,
    *,
    fetcher: Callable[..., Mapping[str, Any]] = fetch_schedule_raw,
    timestamp: datetime | None = None,
) -> BuildSummary:
    """Build canonical regular-season, history, audit, and manifest artifacts."""
    started = time.monotonic()
    build_time = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    retrieved_at = build_time.isoformat()
    season_list = tuple(sorted({int(season) for season in seasons}))
    if not season_list:
        raise ScheduleBuildError("At least one season is required")

    payloads: list[Mapping[str, Any]] = []
    for season in season_list:
        start, end = _season_dates(season)
        payloads.append(fetcher(start, end, game_types=["R"]))

    history = normalize_schedule_history(payloads, retrieved_at_utc=retrieved_at)
    audit = build_schedule_change_audit(history)
    canonical = normalize_schedule(payloads, retrieved_at_utc=retrieved_at)
    validation = _validate_regular_only(canonical)
    validation.update(schedule_history_metrics(history, audit))
    validation["validated_at_utc"] = retrieved_at

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "schedule_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if validation["status"] != "passed":
        raise ScheduleBuildError("; ".join(validation["errors"]))

    canonical_path = output / "canonical_schedule.parquet"
    history_path = output / "schedule_history.parquet"
    audit_path = output / "rescheduled_games_audit.csv"
    _write_parquet(canonical, canonical_path)

    history_columns = list(canonical[0]) + list(HISTORY_EXTRA_FIELDS) if canonical else None
    _write_history_parquet(history, history_path, columns=history_columns)
    _write_audit(audit, audit_path)

    locations = {name: str(output / name) for name in ARTIFACT_NAMES}
    hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in (
            "canonical_schedule.parquet",
            "schedule_history.parquet",
            "rescheduled_games_audit.csv",
        )
    }
    manifest = {
        "builder": "atlas.schedule.schedule_builder_v2",
        "builder_version": "2",
        "required_game_type": "R",
        "seasons": list(season_list),
        "games_processed": len(canonical),
        "schedule_history_rows": len(history),
        "schedule_affected_games": len(audit),
        "artifacts": locations,
        "artifact_hashes": hashes,
        "schedule_history_implemented": True,
        "source": "mlb_stats_api_schedule",
        "git_revision": _git_revision(),
        "manifest_content_hash": None,
    }
    manifest["manifest_content_hash"] = _stable_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    )
    (output / "schedule_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BuildSummary(
        seasons_built=season_list,
        games_processed=len(canonical),
        duplicate_count=validation["duplicate_count"],
        validation_status=validation["status"],
        artifact_locations=locations,
        elapsed_build_time_seconds=time.monotonic() - started,
        timestamp=retrieved_at,
    )
