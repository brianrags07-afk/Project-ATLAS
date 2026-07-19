"""
Historical coverage matrix for 2024/2025/2026, built strictly from
evidence gathered by the repository inventory and dataset profile steps.
No row is marked "complete" or "present" without a cited evidence source;
anything not directly evidenced is "unknown".
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEASONS = (2024, 2025, 2026)

ALLOWED_STATUSES = (
    "complete",
    "partial",
    "missing",
    "unknown",
    "present_but_not_pregame_safe",
)

COVERAGE_ROWS = (
    "published_schedule",
    "game_identifiers",
    "scheduled_first_pitch",
    "final_scores",
    "pitch_by_pitch",
    "plate_appearances",
    "batted_ball_data",
    "starters",
    "bullpen_usage",
    "lineups",
    "injuries",
    "weather",
    "venue",
    "umpire",
    "rest",
    "travel",
    "published_series_context",
    "opening_market",
    "closing_market",
    "team_memories",
    "player_memories",
    "identities",
    "concept_discovery",
    "concept_validation",
    "model_artifacts",
    "frozen_predictions",
    "frozen_pregame_cards",
)

# Maps a coverage row to the dataset_profile "feature_presence" key(s) and/or
# repository inventory focus area(s) that provide direct evidence for it.
ROW_TO_FEATURE_PRESENCE_KEY = {
    "game_identifiers": "game_pk",
    "scheduled_first_pitch": "scheduled_first_pitch",
    "final_scores": "final_outcomes",
    "starters": "starter_information",
    "bullpen_usage": "bullpen_usage",
    "lineups": "lineups",
    "injuries": "injuries",
    "weather": "weather",
    "venue": "venue",
    "umpire": "umpire",
    "rest": "rest",
    "travel": "travel",
    "published_series_context": "published_series_context",
    "opening_market": "market_data",
    "closing_market": "market_data",
}

ROW_TO_FOCUS_AREA = {
    "team_memories": "memories",
    "player_memories": "memories",
    "identities": "identities",
    "concept_discovery": "concepts",
    "concept_validation": "validation",
    "model_artifacts": "prediction",
    "frozen_predictions": "prediction",
    "frozen_pregame_cards": "pregame_snapshots",
}


def _season_has_rows(rows_by_season: dict[str, int], season: int) -> bool:
    return rows_by_season.get(str(season), 0) > 0


def _evidence_for_dataset_row(
    row: str,
    season: int,
    dataset_profiles: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    feature_key = ROW_TO_FEATURE_PRESENCE_KEY.get(row)
    if feature_key is None:
        return "unknown", "no mapped dataset feature-presence key for this row"

    matches = []
    for dataset_name, profile in dataset_profiles.items():
        column = profile.get("feature_presence", {}).get(feature_key)
        has_season_rows = _season_has_rows(profile.get("rows_by_season", {}), season)
        if column and has_season_rows:
            matches.append((dataset_name, column))

    if not matches:
        for dataset_name, profile in dataset_profiles.items():
            if profile.get("feature_presence", {}).get(feature_key):
                return (
                    "unknown",
                    f"column `{profile['feature_presence'][feature_key]}` exists in {dataset_name} "
                    f"but no rows found for season {season}",
                )
        return "missing", f"no dataset column found matching '{feature_key}'"

    dataset_name, column = matches[0]
    classification = dataset_profiles[dataset_name].get("column_classification", {}).get(column, "unknown")
    if classification == "schedule_safe" or feature_key in ("game_pk", "scheduled_first_pitch"):
        return "complete", f"column `{column}` present in {dataset_name} with rows for season {season}"
    if classification == "pregame_possible_but_needs_timestamp_proof":
        return (
            "present_but_not_pregame_safe",
            f"column `{column}` present in {dataset_name} for season {season}, "
            "but has no source/retrieval timestamp proving pregame availability",
        )
    if classification == "postgame_fact":
        return (
            "present_but_not_pregame_safe",
            f"column `{column}` present in {dataset_name} for season {season} but is a postgame fact",
        )
    return "partial", f"column `{column}` present in {dataset_name} for season {season}; classification={classification}"


def _evidence_for_module_row(row: str, season: int, repository_inventory: dict[str, Any]) -> tuple[str, str]:
    focus_area = ROW_TO_FOCUS_AREA.get(row)
    if focus_area is None:
        return "unknown", "no mapped repository focus area for this row"
    modules = repository_inventory.get("focus_area_index", {}).get(focus_area, [])
    if not modules:
        return "missing", f"no repository modules found for focus area '{focus_area}'"
    return (
        "unknown",
        f"module(s) {modules} exist for focus area '{focus_area}', but season-{season} "
        "artifact production was not directly observed by this audit (repo inspection only)",
    )


def build_coverage_matrix(
    dataset_profiles: dict[str, dict[str, Any]],
    repository_inventory: dict[str, Any],
    cloud_inventory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedule_profile = dataset_profiles.get("master_game_database")

    for row_name in COVERAGE_ROWS:
        for season in SEASONS:
            if row_name == "published_schedule":
                if schedule_profile and _season_has_rows(schedule_profile.get("rows_by_season", {}), season):
                    status, evidence = "complete", (
                        f"master_game_database has {schedule_profile['rows_by_season'][str(season)]} "
                        f"rows for season {season}"
                    )
                else:
                    status, evidence = "missing", "no master_game_database rows found for this season"
            elif row_name in ("pitch_by_pitch", "plate_appearances", "batted_ball_data"):
                pitch_profile = dataset_profiles.get("master_pitch_database")
                if pitch_profile and _season_has_rows(pitch_profile.get("pitches_by_season", pitch_profile.get("rows_by_season", {})), season):
                    status = "present_but_not_pregame_safe"
                    evidence = f"master_pitch_database has rows for season {season} (postgame pitch-level facts)"
                else:
                    status, evidence = "missing", "no master_pitch_database rows found for this season"
            elif row_name in ROW_TO_FEATURE_PRESENCE_KEY:
                status, evidence = _evidence_for_dataset_row(row_name, season, dataset_profiles)
            elif row_name in ROW_TO_FOCUS_AREA:
                status, evidence = _evidence_for_module_row(row_name, season, repository_inventory)
            else:
                status, evidence = "unknown", "no evidence source mapped for this row"

            if status not in ALLOWED_STATUSES:
                status = "unknown"

            rows.append({
                "row": row_name,
                "season": season,
                "status": status,
                "evidence": evidence,
            })
    return rows


def write_coverage_matrix(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "historical_coverage_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["row", "season", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# ATLAS Historical Coverage Matrix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Row | Season | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for r in rows:
        evidence = r["evidence"].replace("|", "\\|")
        md_lines.append(f"| {r['row']} | {r['season']} | {r['status']} | {evidence} |")
    md_path = output_dir / "historical_coverage_matrix.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return csv_path, md_path
