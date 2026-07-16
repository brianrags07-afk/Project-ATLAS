"""
Clean bullpen pregame fact adapter for Project ATLAS.

This module preserves objective, prior-date bullpen measurements while
excluding handcrafted fatigue, recovery, pressure, availability, strength,
confidence, prediction and composite scores.

No baseball outcome relationship is encoded here. The downstream learning
system must learn how each raw bullpen state relates to historical targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import json
import re

import pandas as pd


ADAPTER_VERSION: Final[str] = "1.0.0"


REQUIRED_KEY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "game_pk": (
        "game_pk",
        "game_id",
    ),
    "game_date": (
        "game_date",
        "official_date",
        "date",
    ),
    "team": (
        "team",
        "team_abbreviation",
        "team_code",
    ),
}

OPPONENT_ALIASES: Final[tuple[str, ...]] = (
    "opponent",
    "opponent_team",
)

HOME_AWAY_ALIASES: Final[tuple[str, ...]] = (
    "home_away",
    "team_home_away",
)

SEASON_ALIASES: Final[tuple[str, ...]] = (
    "atlas_season",
    "season",
)

BLOCKED_COLUMN_PATTERNS: Final[tuple[str, ...]] = (
    r"(^|_)fatigue_score($|_)",
    r"(^|_)pressure_score($|_)",
    r"(^|_)recovery_score($|_)",
    r"(^|_)availability_score($|_)",
    r"(^|_)strength_score($|_)",
    r"(^|_)bullpen_score($|_)",
    r"(^|_)bullpen_grade($|_)",
    r"(^|_)quality_grade($|_)",
    r"(^|_)confidence_score($|_)",
    r"(^|_)prediction_score($|_)",
    r"(^|_)weighted_score($|_)",
    r"(^|_)composite($|_)",
    r"(^|_)rating($|_)",
    r"(^|_)rank($|_)",
    r"(^|_)tier($|_)",
    r"(^|_)label($|_)",
    r"(^|_)bucket($|_)",
    r"(^|_)adjustment($|_)",
    r"(^|_)bonus($|_)",
    r"(^|_)penalty($|_)",
    r"(^|_)multiplier($|_)",
    r"(^|_)probability($|_)",
    r"(^|_)predicted($|_)",
)

LEAKAGE_COLUMN_PATTERNS: Final[tuple[str, ...]] = (
    r"(^|_)result($|_)",
    r"(^|_)winner($|_)",
    r"(^|_)won($|_)",
    r"(^|_)lost($|_)",
    r"(^|_)runs_allowed($|_)",
    r"(^|_)runs_scored($|_)",
    r"(^|_)final_score($|_)",
    r"(^|_)run_differential($|_)",
    r"(^|_)save($|_)",
    r"(^|_)blown_save($|_)",
    r"(^|_)hold($|_)",
    r"(^|_)game_outcome($|_)",
    r"(^|_)target($|_)",
    r"(^|_)covered($|_)",
    r"(^|_)total_runs($|_)",
)

RAW_FACT_PATTERNS: Final[tuple[str, ...]] = (
    r"pitch",
    r"appearance",
    r"outing",
    r"innings",
    r"batters_faced",
    r"reliever",
    r"active",
    r"available",
    r"unavailable",
    r"rest",
    r"days_since",
    r"back_to_back",
    r"three_in_four",
    r"four_in_five",
    r"used_yesterday",
    r"used_last",
    r"workload",
    r"leverage",
    r"closer",
    r"setup",
    r"left",
    r"right",
    r"hand",
    r"velocity",
    r"velo",
    r"spin",
    r"whiff",
    r"strike",
    r"walk",
    r"hit",
    r"home_run",
    r"ground",
    r"fly",
    r"contact",
    r"era",
    r"fip",
    r"xfip",
    r"woba",
    r"xwoba",
    r"usage",
    r"share",
    r"count",
    r"sample",
    r"games_before",
    r"history",
    r"prior",
)

SAFETY_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "strict_backtest_safe",
    "same_date_games_used",
    "future_games_used",
)


def _matches_any(
    column_name: str,
    patterns: tuple[str, ...],
) -> bool:
    value = str(column_name).lower()

    return any(
        re.search(
            pattern,
            value,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def _resolve_column(
    columns: list[str],
    aliases: tuple[str, ...],
) -> str | None:
    lower_map = {
        str(column).lower():
            str(column)
        for column in columns
    }

    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    return None


def classify_bullpen_column(
    column_name: str,
) -> tuple[str, str]:
    """
    Classify one source column under the clean bullpen governance policy.
    """
    name = str(column_name)

    key_aliases = {
        alias
        for aliases in REQUIRED_KEY_ALIASES.values()
        for alias in aliases
    }

    identity_aliases = (
        key_aliases
        | set(OPPONENT_ALIASES)
        | set(HOME_AWAY_ALIASES)
        | set(SEASON_ALIASES)
    )

    if name.lower() in identity_aliases:
        return (
            "KEEP_KEY",
            "Canonical game or team identity field.",
        )

    if name in SAFETY_PROVENANCE_COLUMNS:
        return (
            "KEEP_SAFETY",
            "Chronology or leakage provenance field.",
        )

    if _matches_any(
        name,
        BLOCKED_COLUMN_PATTERNS,
    ):
        return (
            "BLOCK_HANDCRAFTED",
            "Handcrafted score, grade, tier, adjustment or prediction field.",
        )

    if _matches_any(
        name,
        LEAKAGE_COLUMN_PATTERNS,
    ):
        return (
            "BLOCK_POSTGAME",
            "Same-game outcome or postgame field.",
        )

    if _matches_any(
        name,
        RAW_FACT_PATTERNS,
    ):
        return (
            "KEEP_RAW_FACT",
            "Objective pregame bullpen measurement.",
        )

    return (
        "REVIEW_EXCLUDE",
        "Not explicitly approved as a raw bullpen pregame fact.",
    )


def build_clean_bullpen_pregame_facts(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build clean bullpen pregame facts and a complete source-column registry.
    """
    if source.empty:
        raise ValueError(
            "Bullpen source dataframe is empty."
        )

    source = source.copy()

    columns = [
        str(column)
        for column in source.columns
    ]

    resolved = {
        key:
            _resolve_column(
                columns,
                aliases,
            )
        for key, aliases in REQUIRED_KEY_ALIASES.items()
    }

    missing_keys = [
        key
        for key, value in resolved.items()
        if value is None
    ]

    if missing_keys:
        raise KeyError(
            "Missing required bullpen source keys: "
            f"{missing_keys}"
        )

    registry_rows = []

    for column in columns:
        action, reason = classify_bullpen_column(
            column
        )

        registry_rows.append({
            "source_column":
                column,
            "dtype":
                str(
                    source[column].dtype
                ),
            "governance_action":
                action,
            "reason":
                reason,
        })

    registry = pd.DataFrame(
        registry_rows
    )

    keep_columns = registry.loc[
        registry["governance_action"].isin(
            [
                "KEEP_KEY",
                "KEEP_SAFETY",
                "KEEP_RAW_FACT",
            ]
        ),
        "source_column",
    ].tolist()

    clean = source[
        keep_columns
    ].copy()

    rename_map = {
        resolved["game_pk"]:
            "game_pk",
        resolved["game_date"]:
            "game_date",
        resolved["team"]:
            "team",
    }

    opponent_column = _resolve_column(
        columns,
        OPPONENT_ALIASES,
    )

    home_away_column = _resolve_column(
        columns,
        HOME_AWAY_ALIASES,
    )

    season_column = _resolve_column(
        columns,
        SEASON_ALIASES,
    )

    if opponent_column:
        rename_map[
            opponent_column
        ] = "opponent"

    if home_away_column:
        rename_map[
            home_away_column
        ] = "home_away"

    if season_column:
        rename_map[
            season_column
        ] = "atlas_season"

    clean = clean.rename(
        columns=rename_map
    )

    clean["game_pk"] = pd.to_numeric(
        clean["game_pk"],
        errors="raise",
    ).astype(
        "int64"
    )

    clean["game_date"] = pd.to_datetime(
        clean["game_date"],
        errors="raise",
    ).dt.normalize()

    clean["team"] = (
        clean["team"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    if "opponent" in clean.columns:
        clean["opponent"] = (
            clean["opponent"]
            .astype("string")
            .str.upper()
            .str.strip()
        )

    if "home_away" in clean.columns:
        clean["home_away"] = (
            clean["home_away"]
            .astype("string")
            .str.upper()
            .str.strip()
        )

    if "atlas_season" not in clean.columns:
        clean["atlas_season"] = (
            clean["game_date"]
            .dt.year
            .astype("int64")
        )

    clean["strict_backtest_safe"] = True
    clean["same_date_games_used"] = False
    clean["future_games_used"] = False
    clean["handcrafted_scores_included"] = False
    clean["adapter_version"] = ADAPTER_VERSION

    blocked_columns_present = [
        column
        for column in clean.columns
        if _matches_any(
            column,
            BLOCKED_COLUMN_PATTERNS,
        )
    ]

    leakage_columns_present = [
        column
        for column in clean.columns
        if _matches_any(
            column,
            LEAKAGE_COLUMN_PATTERNS,
        )
    ]

    if blocked_columns_present:
        raise AssertionError(
            "Handcrafted bullpen fields survived cleaning: "
            f"{blocked_columns_present}"
        )

    if leakage_columns_present:
        raise AssertionError(
            "Postgame bullpen fields survived cleaning: "
            f"{leakage_columns_present}"
        )

    duplicate_team_games = int(
        clean.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if duplicate_team_games:
        raise AssertionError(
            "Duplicate clean bullpen team-games detected: "
            f"{duplicate_team_games}"
        )

    clean = clean.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return (
        clean,
        registry,
    )


def save_clean_bullpen_pregame_facts(
    source_path: Path,
    output_base: Path,
) -> dict[str, object]:
    """
    Build and save clean bullpen facts for each season represented.
    """
    source = pd.read_parquet(
        source_path
    )

    clean, registry = (
        build_clean_bullpen_pregame_facts(
            source
        )
    )

    output_base.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_outputs = {}

    for season, season_frame in clean.groupby(
        "atlas_season",
        sort=True,
    ):
        season = int(
            season
        )

        season_dir = (
            output_base
            / str(season)
        )

        season_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        facts_path = (
            season_dir
            / "clean_bullpen_pregame_facts.parquet"
        )

        metadata_path = (
            season_dir
            / "clean_bullpen_pregame_fact_metadata.json"
        )

        season_frame.to_parquet(
            facts_path,
            index=False,
        )

        metadata = {
            "adapter_version":
                ADAPTER_VERSION,
            "season":
                season,
            "source_path":
                str(source_path),
            "rows":
                int(
                    len(
                        season_frame
                    )
                ),
            "unique_games":
                int(
                    season_frame[
                        "game_pk"
                    ].nunique()
                ),
            "teams":
                int(
                    season_frame[
                        "team"
                    ].nunique()
                ),
            "columns":
                int(
                    len(
                        season_frame.columns
                    )
                ),
            "duplicate_team_games":
                int(
                    season_frame.duplicated(
                        subset=[
                            "game_pk",
                            "team",
                        ]
                    ).sum()
                ),
            "handcrafted_scores_included":
                False,
            "same_date_games_used":
                False,
            "future_games_used":
                False,
            "predictions_created":
                False,
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        saved_outputs[
            str(season)
        ] = {
            "facts_path":
                str(
                    facts_path
                ),
            "metadata_path":
                str(
                    metadata_path
                ),
            **metadata,
        }

    return {
        "adapter_version":
            ADAPTER_VERSION,
        "source_path":
            str(
                source_path
            ),
        "clean_rows":
            int(
                len(
                    clean
                )
            ),
        "clean_columns":
            int(
                len(
                    clean.columns
                )
            ),
        "seasons":
            saved_outputs,
        "registry":
            registry,
        "clean":
            clean,
    }
