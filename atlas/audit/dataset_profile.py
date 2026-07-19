"""
Dataset profiling helpers for the ATLAS historical readiness audit.

Profiles the four known master datasets that are already downloaded to
local disk by the calling workflow (read-only ``gcloud storage cp``, never
a write/upload). This module never guesses column meaning: a column is
only classified when there is direct evidence (name/dtype/value pattern)
for that classification, otherwise it is reported as ``"unknown"``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# Column classification evidence tables.
#
# These are *not* invented columns -- they are keyword patterns used only to
# classify columns that are actually present in a given dataset. A column
# that does not match any pattern is left as "unknown".
# --------------------------------------------------------------------------

IDENTIFIER_PATTERNS = ("game_pk", "_id", "game_id", "player_id", "pitcher_id", "batter_id", "team_id")
SCHEDULE_SAFE_PATTERNS = (
    "game_date",
    "scheduled",
    "first_pitch",
    "home_team",
    "away_team",
    "venue",
    "double_header",
    "doubleheader",
    "game_number",
    "series",
)
POSTGAME_FACT_PATTERNS = (
    "final_score",
    "score",
    "result",
    "outcome",
    "win",
    "loss",
    "runs_scored",
    "runs_allowed",
    "events",
    "description",
    "pitch_type",
    "release_speed",
    "launch_speed",
    "launch_angle",
    "hit_",
    "rbi",
)
PREDICTION_TARGET_PATTERNS = ("target", "label", "predicted", "prediction")
NEEDS_TIMESTAMP_PROOF_PATTERNS = (
    "starter",
    "lineup",
    "bullpen",
    "injury",
    "injuries",
    "weather",
    "umpire",
    "rest_days",
    "travel",
    "odds",
    "line",
    "market",
)

HIGH_NULL_THRESHOLD_PERCENT = 50.0

SEASON_COLUMN_CANDIDATES = ("season", "game_year", "year")
GAME_DATE_COLUMN_CANDIDATES = ("game_date", "date", "game_datetime")
GAME_PK_COLUMN_CANDIDATES = ("game_pk", "game_id")

FEATURE_PRESENCE_CHECKS: dict[str, tuple[str, ...]] = {
    "game_pk": GAME_PK_COLUMN_CANDIDATES,
    "game_date": GAME_DATE_COLUMN_CANDIDATES,
    "scheduled_first_pitch": ("scheduled_first_pitch", "first_pitch", "game_datetime", "scheduled_start"),
    "home_away_teams": ("home_team", "away_team"),
    "final_outcomes": ("home_score", "away_score", "final_score", "winning_team", "result"),
    "pitch_ordering": ("pitch_number", "at_bat_number", "inning"),
    "starter_information": ("starter", "starting_pitcher"),
    "bullpen_usage": ("bullpen",),
    "lineups": ("lineup", "batting_order"),
    "injuries": ("injury", "injuries", "il_status"),
    "weather": ("weather", "temperature", "wind"),
    "venue": ("venue",),
    "umpire": ("umpire",),
    "rest": ("rest_days", "days_rest"),
    "travel": ("travel",),
    "published_series_context": ("series_game_number", "series_length"),
    "market_data": ("odds", "line", "moneyline", "spread", "total_line", "market"),
    "source_timestamps": ("retrieved_at", "source_timestamp", "ingested_at", "updated_at"),
}


def classify_column(column: str) -> str:
    lowered = column.lower()
    if any(p in lowered for p in IDENTIFIER_PATTERNS):
        return "identifier"
    if any(p in lowered for p in PREDICTION_TARGET_PATTERNS):
        return "prediction_target"
    if any(p in lowered for p in NEEDS_TIMESTAMP_PROOF_PATTERNS):
        return "pregame_possible_but_needs_timestamp_proof"
    if any(p in lowered for p in POSTGAME_FACT_PATTERNS):
        return "postgame_fact"
    if any(p in lowered for p in SCHEDULE_SAFE_PATTERNS):
        return "schedule_safe"
    return "unknown"


def _find_first_matching_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:
        for lower_col, original in lowered.items():
            if candidate in lower_col:
                return original
    return None


def _feature_presence(columns: list[str]) -> dict[str, str | None]:
    return {
        feature: _find_first_matching_column(columns, candidates)
        for feature, candidates in FEATURE_PRESENCE_CHECKS.items()
    }


def detect_season_column(df: pd.DataFrame) -> str | None:
    return _find_first_matching_column(list(df.columns), SEASON_COLUMN_CANDIDATES)


def detect_game_date_column(df: pd.DataFrame) -> str | None:
    return _find_first_matching_column(list(df.columns), GAME_DATE_COLUMN_CANDIDATES)


def detect_game_pk_column(df: pd.DataFrame) -> str | None:
    return _find_first_matching_column(list(df.columns), GAME_PK_COLUMN_CANDIDATES)


def derive_season_from_date(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Fallback season derivation used only when no explicit season/year
    column is present. This intentionally uses the calendar year of
    ``game_date`` as the season, which is a simplification: MLB
    postseason games can occur in a later calendar year than the
    season's opening day. When a dataset has an explicit season column,
    that column is used instead (see ``rows_by_season``/
    ``unique_games_by_season``) and this fallback is not invoked."""
    parsed = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    return parsed.dt.year


def rows_by_season(df: pd.DataFrame) -> dict[str, int]:
    season_col = detect_season_column(df)
    if season_col is None:
        date_col = detect_game_date_column(df)
        if date_col is None:
            return {}
        seasons = derive_season_from_date(df, date_col)
    else:
        seasons = pd.to_numeric(df[season_col], errors="coerce")
    counts = seasons.dropna().astype(int).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def unique_games_by_season(df: pd.DataFrame) -> dict[str, int]:
    game_pk_col = detect_game_pk_column(df)
    season_col = detect_season_column(df)
    if game_pk_col is None:
        return {}
    if season_col is None:
        date_col = detect_game_date_column(df)
        if date_col is None:
            return {}
        seasons = derive_season_from_date(df, date_col)
    else:
        seasons = pd.to_numeric(df[season_col], errors="coerce")
    tmp = pd.DataFrame({"season": seasons, "game_pk": df[game_pk_col]}).dropna(subset=["season"])
    tmp["season"] = tmp["season"].astype(int)
    counts = tmp.groupby("season")["game_pk"].nunique().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def detect_likely_primary_key(df: pd.DataFrame, candidate_keys: list[list[str]]) -> tuple[list[str] | None, int]:
    """Return the first candidate key combination (from most to least
    specific) that is fully present in the dataframe, along with the count
    of duplicate rows for that key. If none of the candidates are present,
    returns (None, -1) rather than guessing a key."""
    for key in candidate_keys:
        if all(col in df.columns for col in key):
            dup_count = int(df.duplicated(subset=key).sum())
            return key, dup_count
    return None, -1


def duplicate_columns(df: pd.DataFrame) -> list[str]:
    seen: dict[str, int] = {}
    duplicates = []
    for col in df.columns:
        seen[col] = seen.get(col, 0) + 1
        if seen[col] > 1:
            duplicates.append(col)
    return duplicates


def null_percentages(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {col: 0.0 for col in df.columns}
    return {col: round(float(df[col].isna().mean() * 100), 4) for col in df.columns}


def teams_present(df: pd.DataFrame) -> list[str]:
    teams: set[str] = set()
    for col_candidate in ("home_team", "away_team", "team"):
        col = _find_first_matching_column(list(df.columns), (col_candidate,))
        if col:
            teams.update(str(v) for v in df[col].dropna().unique())
    return sorted(teams)


def profile_dataframe(
    df: pd.DataFrame,
    cloud_path: str,
    local_size_bytes: int,
    candidate_keys: list[list[str]],
) -> dict[str, Any]:
    columns = list(df.columns)
    date_col = detect_game_date_column(df)
    min_date = max_date = None
    if date_col is not None:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if parsed_dates.notna().any():
            min_date = parsed_dates.min().isoformat()
            max_date = parsed_dates.max().isoformat()

    likely_key, dup_count = detect_likely_primary_key(df, candidate_keys)
    season_counts = rows_by_season(df)

    return {
        "cloud_path": cloud_path,
        "local_size_bytes": local_size_bytes,
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "table_grain_hint": ", ".join(likely_key) if likely_key else "unknown",
        "likely_primary_key": likely_key,
        "duplicate_key_count": dup_count,
        "min_game_date": min_date,
        "max_game_date": max_date,
        "seasons_present": sorted(season_counts.keys(), key=int) if season_counts else [],
        "rows_by_season": season_counts,
        "unique_games_by_season": unique_games_by_season(df),
        "teams_present": teams_present(df),
        "null_percentages": null_percentages(df),
        "duplicate_columns": duplicate_columns(df),
        "feature_presence": _feature_presence(columns),
        "column_classification": {col: classify_column(col) for col in columns},
    }


def profile_master_game_database(df: pd.DataFrame, cloud_path: str, local_size_bytes: int) -> dict[str, Any]:
    return profile_dataframe(
        df, cloud_path, local_size_bytes,
        candidate_keys=[["game_pk"], ["game_id"]],
    )


def profile_team_game_state(df: pd.DataFrame, cloud_path: str, local_size_bytes: int) -> dict[str, Any]:
    profile = profile_dataframe(
        df, cloud_path, local_size_bytes,
        candidate_keys=[["game_pk", "team"], ["game_pk", "team_id"], ["game_id", "team"]],
    )
    profile["grain"] = "game_pk + team (one row per team per game)" if profile["likely_primary_key"] else "unknown"
    return profile


def profile_master_pitch_database(df: pd.DataFrame, cloud_path: str, local_size_bytes: int) -> dict[str, Any]:
    profile = profile_dataframe(
        df, cloud_path, local_size_bytes,
        candidate_keys=[
            ["game_pk", "at_bat_number", "pitch_number"],
            ["game_pk", "inning", "at_bat_number", "pitch_number"],
        ],
    )
    columns = list(df.columns)
    chronology_columns = ["inning", "at_bat_number", "pitch_number"]
    profile["chronology_reconstructable"] = all(
        _find_first_matching_column(columns, (c,)) for c in chronology_columns
    )
    profile["pitches_by_season"] = profile["rows_by_season"]
    profile["unique_games_by_season"] = unique_games_by_season(df)
    return profile


def profile_metadata_json(metadata: dict[str, Any], actual_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare the metadata JSON's claims about row counts / date ranges
    against the actual profiled files. Only compares fields that are
    actually present in the metadata document -- never invents fields."""
    comparisons = []
    for dataset_name, claimed in metadata.items():
        if not isinstance(claimed, dict):
            continue
        actual = actual_profiles.get(dataset_name)
        if actual is None:
            comparisons.append({
                "dataset": dataset_name,
                "status": "no_matching_profiled_dataset",
            })
            continue
        entry = {"dataset": dataset_name}
        if "row_count" in claimed:
            entry["claimed_row_count"] = claimed["row_count"]
            entry["actual_row_count"] = actual["row_count"]
            entry["row_count_match"] = claimed["row_count"] == actual["row_count"]
        comparisons.append(entry)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_metadata": metadata,
        "comparisons": comparisons,
    }


def write_dataset_profile_reports(
    profiles: dict[str, dict[str, Any]],
    metadata_comparison: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": profiles,
        "metadata_comparison": metadata_comparison,
    }
    json_path = output_dir / "dataset_profile.json"
    json_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")

    md_lines = ["# ATLAS Dataset Profile", ""]
    for name, profile in profiles.items():
        md_lines.append(f"## {name}")
        md_lines.append(f"- cloud_path: `{profile.get('cloud_path')}`")
        md_lines.append(f"- rows: {profile.get('row_count')}, columns: {profile.get('column_count')}")
        md_lines.append(f"- likely_primary_key: {profile.get('likely_primary_key')}")
        md_lines.append(f"- duplicate_key_count: {profile.get('duplicate_key_count')}")
        md_lines.append(f"- date range: {profile.get('min_game_date')} -> {profile.get('max_game_date')}")
        md_lines.append(f"- seasons_present: {profile.get('seasons_present')}")
        md_lines.append(f"- rows_by_season: {profile.get('rows_by_season')}")
        md_lines.append(f"- unique_games_by_season: {profile.get('unique_games_by_season')}")
        md_lines.append(f"- duplicate_columns: {profile.get('duplicate_columns')}")
        md_lines.append("")
    md_path = output_dir / "dataset_profile.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = output_dir / "column_classification.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("dataset,column,classification,null_percentage,dtype\n")
        for name, profile in profiles.items():
            for col in profile.get("columns", []):
                classification = profile.get("column_classification", {}).get(col, "unknown")
                null_pct = profile.get("null_percentages", {}).get(col, "")
                dtype = profile.get("dtypes", {}).get(col, "")
                fh.write(f'"{name}","{col}","{classification}","{null_pct}","{dtype}"\n')

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "findings": [
            {
                "dataset": name,
                "duplicate_key_count": profile.get("duplicate_key_count"),
                "duplicate_columns": profile.get("duplicate_columns"),
                "high_null_columns": [
                    col
                    for col, pct in profile.get("null_percentages", {}).items()
                    if pct is not None and pct > HIGH_NULL_THRESHOLD_PERCENT
                ],
                "unknown_columns": [
                    col
                    for col, cls in profile.get("column_classification", {}).items()
                    if cls == "unknown"
                ],
            }
            for name, profile in profiles.items()
        ],
    }
    findings_path = output_dir / "data_quality_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    return {
        "dataset_profile_json": json_path,
        "dataset_profile_md": md_path,
        "column_classification_csv": csv_path,
        "data_quality_findings_json": findings_path,
    }
