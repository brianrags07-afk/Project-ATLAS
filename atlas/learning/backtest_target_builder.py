
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR, MASTER_DIR


TARGET_ENGINE_VERSION = "1.0.0"

MASTER_GAME_PATH = (
    MASTER_DIR
    / "master_game_database.parquet"
)

ANOMALY_REGISTRY_PATH = (
    DATA_DIR
    / "validation"
    / "anomalies"
    / "game_anomaly_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "backtest"
    / "targets"
)

GAME_TARGET_PATH = (
    OUTPUT_DIR
    / "game_targets.parquet"
)

TEAM_GAME_TARGET_PATH = (
    OUTPUT_DIR
    / "team_game_targets.parquet"
)

TARGET_METADATA_PATH = (
    OUTPUT_DIR
    / "target_builder_metadata.json"
)


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(destination)


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary.replace(destination)


def _load_source_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not MASTER_GAME_PATH.exists():
        raise FileNotFoundError(
            f"Missing master game database: {MASTER_GAME_PATH}"
        )

    if not ANOMALY_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Missing anomaly registry: {ANOMALY_REGISTRY_PATH}"
        )

    games = pd.read_parquet(
        MASTER_GAME_PATH
    )

    anomalies = pd.read_parquet(
        ANOMALY_REGISTRY_PATH
    )

    return games, anomalies


def _season_column(
    games: pd.DataFrame,
) -> str:
    for column in [
        "atlas_season",
        "game_year",
        "season",
    ]:
        if column in games.columns:
            return column

    raise KeyError(
        "No season column found in master game database."
    )


def _validate_required_columns(
    games: pd.DataFrame,
) -> None:
    required = {
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing = required - set(
        games.columns
    )

    if missing:
        raise KeyError(
            f"Master game database missing columns: "
            f"{sorted(missing)}"
        )


def build_game_targets(
    games: pd.DataFrame,
    anomaly_registry: pd.DataFrame,
) -> pd.DataFrame:
    _validate_required_columns(
        games
    )

    season_column = _season_column(
        games
    )

    safe_game_pks = set(
        pd.to_numeric(
            anomaly_registry.loc[
                anomaly_registry[
                    "strict_backtest_safe"
                ].fillna(False),
                "game_pk",
            ],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    targets = games[
        games["game_pk"].isin(
            safe_game_pks
        )
    ][
        [
            "game_pk",
            "game_date",
            season_column,
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    targets = targets.rename(
        columns={
            season_column: "atlas_season",
        }
    )

    targets["game_pk"] = pd.to_numeric(
        targets["game_pk"],
        errors="raise",
    ).astype("int64")

    targets["game_date"] = pd.to_datetime(
        targets["game_date"],
        errors="raise",
    ).dt.normalize()

    targets["atlas_season"] = pd.to_numeric(
        targets["atlas_season"],
        errors="raise",
    ).astype("int64")

    targets["home_score"] = pd.to_numeric(
        targets["home_score"],
        errors="raise",
    ).astype("int64")

    targets["away_score"] = pd.to_numeric(
        targets["away_score"],
        errors="raise",
    ).astype("int64")

    targets["game_total_runs"] = (
        targets["home_score"]
        + targets["away_score"]
    )

    targets["run_margin"] = (
        targets["home_score"]
        - targets["away_score"]
    )

    targets["home_win"] = (
        targets["home_score"]
        > targets["away_score"]
    )

    targets["away_win"] = (
        targets["away_score"]
        > targets["home_score"]
    )

    targets["one_run_game"] = (
        targets["run_margin"]
        .abs()
        .eq(1)
    )

    targets["extra_inning_or_tied_regulation"] = (
        targets["run_margin"]
        .ne(0)
    )

    # Explicit total thresholds.
    targets["game_total_6_or_less"] = (
        targets["game_total_runs"] <= 6
    )

    targets["game_total_7_or_less"] = (
        targets["game_total_runs"] <= 7
    )

    targets["game_total_8_or_less"] = (
        targets["game_total_runs"] <= 8
    )

    targets["game_total_9_plus"] = (
        targets["game_total_runs"] >= 9
    )

    targets["game_total_10_plus"] = (
        targets["game_total_runs"] >= 10
    )

    # Equivalent to going over a statistical 10.5 threshold.
    targets["game_total_10_5_plus"] = (
        targets["game_total_runs"] >= 11
    )

    targets["game_total_12_plus"] = (
        targets["game_total_runs"] >= 12
    )

    targets["game_total_15_plus"] = (
        targets["game_total_runs"] >= 15
    )

    targets["game_total_17_plus"] = (
        targets["game_total_runs"] >= 17
    )

    targets["both_teams_scored_4_plus"] = (
        targets["home_score"].ge(4)
        & targets["away_score"].ge(4)
    )

    targets["both_teams_scored_5_plus"] = (
        targets["home_score"].ge(5)
        & targets["away_score"].ge(5)
    )

    targets["either_team_scored_8_plus"] = (
        targets["home_score"].ge(8)
        | targets["away_score"].ge(8)
    )

    targets["either_team_scored_10_plus"] = (
        targets["home_score"].ge(10)
        | targets["away_score"].ge(10)
    )

    targets["strict_backtest_safe"] = True
    targets["market_line_used"] = False
    targets["target_engine_version"] = (
        TARGET_ENGINE_VERSION
    )

    return targets.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_team_game_targets(
    game_targets: pd.DataFrame,
) -> pd.DataFrame:
    home = game_targets[
        [
            "game_pk",
            "game_date",
            "atlas_season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_total_runs",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "runs_scored",
            "away_score": "runs_allowed",
        }
    )

    home["home_away"] = "HOME"

    away = game_targets[
        [
            "game_pk",
            "game_date",
            "atlas_season",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "game_total_runs",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "runs_scored",
            "home_score": "runs_allowed",
        }
    )

    away["home_away"] = "AWAY"

    targets = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    targets["run_differential"] = (
        targets["runs_scored"]
        - targets["runs_allowed"]
    )

    targets["won"] = (
        targets["runs_scored"]
        > targets["runs_allowed"]
    )

    targets["lost"] = (
        targets["runs_scored"]
        < targets["runs_allowed"]
    )

    targets["team_scored_0"] = (
        targets["runs_scored"].eq(0)
    )

    targets["team_scored_2_or_less"] = (
        targets["runs_scored"] <= 2
    )

    targets["team_scored_3_or_less"] = (
        targets["runs_scored"] <= 3
    )

    targets["team_scored_exactly_4"] = (
        targets["runs_scored"].eq(4)
    )

    targets["team_scored_5_plus"] = (
        targets["runs_scored"] >= 5
    )

    targets["team_scored_6_plus"] = (
        targets["runs_scored"] >= 6
    )

    targets["team_scored_8_plus"] = (
        targets["runs_scored"] >= 8
    )

    targets["team_scored_10_plus"] = (
        targets["runs_scored"] >= 10
    )

    targets["team_allowed_2_or_less"] = (
        targets["runs_allowed"] <= 2
    )

    targets["team_allowed_3_or_less"] = (
        targets["runs_allowed"] <= 3
    )

    targets["team_allowed_5_plus"] = (
        targets["runs_allowed"] >= 5
    )

    targets["team_allowed_8_plus"] = (
        targets["runs_allowed"] >= 8
    )

    targets["one_run_game"] = (
        targets["run_differential"]
        .abs()
        .eq(1)
    )

    targets["game_total_7_or_less"] = (
        targets["game_total_runs"] <= 7
    )

    targets["game_total_9_plus"] = (
        targets["game_total_runs"] >= 9
    )

    targets["game_total_10_5_plus"] = (
        targets["game_total_runs"] >= 11
    )

    targets["game_total_12_plus"] = (
        targets["game_total_runs"] >= 12
    )

    targets["game_total_15_plus"] = (
        targets["game_total_runs"] >= 15
    )

    targets["game_total_17_plus"] = (
        targets["game_total_runs"] >= 17
    )

    targets["strict_backtest_safe"] = True
    targets["market_team_total_used"] = False
    targets["market_game_total_used"] = False
    targets["target_engine_version"] = (
        TARGET_ENGINE_VERSION
    )

    return targets.sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)


def validate_targets(
    game_targets: pd.DataFrame,
    team_targets: pd.DataFrame,
    anomaly_registry: pd.DataFrame,
) -> dict[str, Any]:
    safe_games = int(
        anomaly_registry[
            "strict_backtest_safe"
        ].fillna(False).sum()
    )

    expected_team_rows = (
        safe_games * 2
    )

    duplicate_games = int(
        game_targets.duplicated(
            subset=["game_pk"]
        ).sum()
    )

    duplicate_team_games = int(
        team_targets.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if len(game_targets) != safe_games:
        raise AssertionError(
            f"Expected {safe_games:,} game targets; "
            f"found {len(game_targets):,}."
        )

    if len(team_targets) != expected_team_rows:
        raise AssertionError(
            f"Expected {expected_team_rows:,} team targets; "
            f"found {len(team_targets):,}."
        )

    if duplicate_games:
        raise AssertionError(
            f"Found {duplicate_games} duplicate game targets."
        )

    if duplicate_team_games:
        raise AssertionError(
            f"Found {duplicate_team_games} duplicate team-game targets."
        )

    if not (
        team_targets.groupby(
            "game_pk"
        ).size().eq(2).all()
    ):
        raise AssertionError(
            "Every game must have exactly two team-target rows."
        )

    return {
        "safe_games": safe_games,
        "game_target_rows": int(
            len(game_targets)
        ),
        "team_game_target_rows": int(
            len(team_targets)
        ),
        "duplicate_games": duplicate_games,
        "duplicate_team_games": duplicate_team_games,
        "teams": int(
            team_targets["team"].nunique()
        ),
        "seasons": sorted(
            int(value)
            for value in team_targets[
                "atlas_season"
            ].dropna().unique()
        ),
        "extreme_total_counts": {
            "10_5_plus": int(
                game_targets[
                    "game_total_10_5_plus"
                ].sum()
            ),
            "12_plus": int(
                game_targets[
                    "game_total_12_plus"
                ].sum()
            ),
            "15_plus": int(
                game_targets[
                    "game_total_15_plus"
                ].sum()
            ),
            "17_plus": int(
                game_targets[
                    "game_total_17_plus"
                ].sum()
            ),
        },
    }


def run_backtest_target_builder() -> dict[str, Any]:
    games, anomaly_registry = (
        _load_source_tables()
    )

    game_targets = build_game_targets(
        games=games,
        anomaly_registry=anomaly_registry,
    )

    team_targets = (
        build_team_game_targets(
            game_targets
        )
    )

    validation = validate_targets(
        game_targets=game_targets,
        team_targets=team_targets,
        anomaly_registry=anomaly_registry,
    )

    _atomic_parquet_write(
        game_targets,
        GAME_TARGET_PATH,
    )

    _atomic_parquet_write(
        team_targets,
        TEAM_GAME_TARGET_PATH,
    )

    metadata = {
        "engine": (
            "ATLAS Backtest Target Builder"
        ),
        "engine_version": (
            TARGET_ENGINE_VERSION
        ),
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "validation": validation,
        "outputs": {
            "game_targets": str(
                GAME_TARGET_PATH
            ),
            "team_game_targets": str(
                TEAM_GAME_TARGET_PATH
            ),
        },
        "target_policy": {
            "pregame_features_modified": False,
            "targets_stored_separately": True,
            "market_lines_used": False,
            "team_total_market_targets_pending": True,
            "league_wide_targets_available": True,
            "team_local_targets_available": True,
        },
    }

    _atomic_json_write(
        metadata,
        TARGET_METADATA_PATH,
    )

    print("=" * 76)
    print("ATLAS BACKTEST TARGET BUILDER")
    print("=" * 76)
    print(
        f"Safe Games............... "
        f"{validation['safe_games']:,}"
    )
    print(
        f"Game Target Rows......... "
        f"{validation['game_target_rows']:,}"
    )
    print(
        f"Team-Game Target Rows.... "
        f"{validation['team_game_target_rows']:,}"
    )
    print(
        f"Teams.................... "
        f"{validation['teams']:,}"
    )
    print(
        f"Seasons.................. "
        f"{validation['seasons']}"
    )
    print(
        "Extreme Totals:"
    )
    print(
        f"  11+ runs............... "
        f"{validation['extreme_total_counts']['10_5_plus']:,}"
    )
    print(
        f"  12+ runs............... "
        f"{validation['extreme_total_counts']['12_plus']:,}"
    )
    print(
        f"  15+ runs............... "
        f"{validation['extreme_total_counts']['15_plus']:,}"
    )
    print(
        f"  17+ runs............... "
        f"{validation['extreme_total_counts']['17_plus']:,}"
    )
    print(
        f"Duplicate Games.......... "
        f"{validation['duplicate_games']:,}"
    )
    print(
        f"Duplicate Team-Games..... "
        f"{validation['duplicate_team_games']:,}"
    )
    print("=" * 76)

    return metadata
