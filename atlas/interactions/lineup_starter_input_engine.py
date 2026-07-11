
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"

HISTORICAL_LINEUP_PATH = (
    DATA_DIR
    / "history"
    / "lineups"
    / "historical_starting_lineups.parquet"
)

BATTER_SNAPSHOT_PATH = (
    DATA_DIR
    / "pregame"
    / "snapshots"
    / "batter_pregame_snapshots.parquet"
)

PITCHER_SNAPSHOT_PATH = (
    DATA_DIR
    / "pregame"
    / "snapshots"
    / "pitcher_pregame_snapshots.parquet"
)

ANOMALY_REGISTRY_PATH = (
    DATA_DIR
    / "validation"
    / "anomalies"
    / "game_anomaly_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "pregame"
    / "interactions"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "lineup_starter_inputs.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "lineup_starter_inputs_metadata.json"
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


def _load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_parquet(path)


def _safe_mean(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    available = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    if not available:
        return pd.Series(
            np.nan,
            index=dataframe.index,
            dtype="float64",
        )

    return dataframe[available].mean(
        axis=1,
        skipna=True,
    )


def _safe_sum(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    available = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    if not available:
        return pd.Series(
            0.0,
            index=dataframe.index,
            dtype="float64",
        )

    return dataframe[available].sum(
        axis=1,
        skipna=True,
    )


def _snapshot_feature_columns(
    snapshots: pd.DataFrame,
    id_column: str,
) -> list[str]:
    excluded = {
        "game_pk",
        "game_date",
        "atlas_season",
        id_column,
        "batting_team",
        "pitching_team",
        "batter_home_away",
        "pitcher_home_away",
        "pregame_snapshot_safe",
        "current_date_games_excluded",
        "current_game_excluded",
        "future_games_excluded",
        "snapshot_builder_version",
    }

    return [
        column
        for column in snapshots.columns
        if column not in excluded
    ]


def _join_batting_slot(
    interaction: pd.DataFrame,
    batter_snapshots: pd.DataFrame,
    position: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    player_column = (
        f"batting_order_{position}_player_id"
    )

    slot_prefix = f"slot_{position}_"

    slot = batter_snapshots[
        [
            "game_pk",
            "player_id",
        ]
        + feature_columns
    ].copy()

    rename_map = {
        "player_id": player_column,
    }

    rename_map.update({
        column: f"{slot_prefix}{column}"
        for column in feature_columns
    })

    slot = slot.rename(
        columns=rename_map
    )

    return interaction.merge(
        slot,
        on=[
            "game_pk",
            player_column,
        ],
        how="left",
        validate="one_to_one",
    )


def _add_lineup_aggregates(
    dataframe: pd.DataFrame,
    batter_feature_columns: list[str],
) -> pd.DataFrame:
    rate_suffixes = [
        "hit_rate_per_pa",
        "home_run_rate_per_pa",
        "walk_rate_per_pa",
        "strikeout_rate_per_pa",
        "swing_pct",
        "whiff_pct_per_swing",
        "called_strike_pct",
        "ball_pct",
        "chase_pct",
        "hard_hit_pct",
        "avg_exit_velocity",
        "avg_launch_angle",
    ]

    total_suffixes = [
        "games",
        "pitches_seen",
        "plate_appearances",
        "hits",
        "home_runs",
        "walks",
        "strikeouts",
        "swings",
        "whiffs",
        "called_strikes",
        "balls_seen",
        "out_zone_pitches",
        "chase_swings",
        "batted_balls",
        "hard_hit_balls",
    ]

    for scope in [
        "career_prior",
        "season_prior",
    ]:
        for suffix in rate_suffixes:
            source_name = f"{scope}_{suffix}"

            slot_columns = [
                f"slot_{position}_{source_name}"
                for position in range(1, 10)
                if f"slot_{position}_{source_name}"
                in dataframe.columns
            ]

            if slot_columns:
                dataframe[
                    f"lineup_{source_name}_mean"
                ] = _safe_mean(
                    dataframe,
                    slot_columns,
                )

                dataframe[
                    f"lineup_{source_name}_min"
                ] = dataframe[
                    slot_columns
                ].min(
                    axis=1,
                    skipna=True,
                )

                dataframe[
                    f"lineup_{source_name}_max"
                ] = dataframe[
                    slot_columns
                ].max(
                    axis=1,
                    skipna=True,
                )

        for suffix in total_suffixes:
            source_name = f"{scope}_{suffix}"

            slot_columns = [
                f"slot_{position}_{source_name}"
                for position in range(1, 10)
                if f"slot_{position}_{source_name}"
                in dataframe.columns
            ]

            if slot_columns:
                dataframe[
                    f"lineup_{source_name}_sum"
                ] = _safe_sum(
                    dataframe,
                    slot_columns,
                )

    season_pa_columns = [
        f"slot_{position}_season_prior_plate_appearances"
        for position in range(1, 10)
        if (
            f"slot_{position}_season_prior_plate_appearances"
            in dataframe.columns
        )
    ]

    career_pa_columns = [
        f"slot_{position}_career_prior_plate_appearances"
        for position in range(1, 10)
        if (
            f"slot_{position}_career_prior_plate_appearances"
            in dataframe.columns
        )
    ]

    if season_pa_columns:
        dataframe[
            "lineup_players_with_season_history"
        ] = (
            dataframe[season_pa_columns]
            .gt(0)
            .sum(axis=1)
            .astype("int8")
        )

    if career_pa_columns:
        dataframe[
            "lineup_players_with_career_history"
        ] = (
            dataframe[career_pa_columns]
            .gt(0)
            .sum(axis=1)
            .astype("int8")
        )

    return dataframe


def build_lineup_starter_inputs(
    lineups: pd.DataFrame,
    batter_snapshots: pd.DataFrame,
    pitcher_snapshots: pd.DataFrame,
    anomaly_registry: pd.DataFrame,
) -> pd.DataFrame:
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

    interaction = lineups[
        lineups["game_pk"].isin(
            safe_game_pks
        )
    ].copy()

    interaction["game_date"] = pd.to_datetime(
        interaction["game_date"],
        errors="coerce",
    ).dt.normalize()

    batter_snapshots = (
        batter_snapshots.copy()
    )

    pitcher_snapshots = (
        pitcher_snapshots.copy()
    )

    batter_snapshots["game_date"] = (
        pd.to_datetime(
            batter_snapshots["game_date"],
            errors="coerce",
        ).dt.normalize()
    )

    pitcher_snapshots["game_date"] = (
        pd.to_datetime(
            pitcher_snapshots["game_date"],
            errors="coerce",
        ).dt.normalize()
    )

    batter_feature_columns = (
        _snapshot_feature_columns(
            batter_snapshots,
            "player_id",
        )
    )

    pitcher_feature_columns = (
        _snapshot_feature_columns(
            pitcher_snapshots,
            "pitcher_id",
        )
    )

    for position in range(1, 10):
        interaction = _join_batting_slot(
            interaction=interaction,
            batter_snapshots=batter_snapshots,
            position=position,
            feature_columns=batter_feature_columns,
        )

    starter = pitcher_snapshots[
        [
            "game_pk",
            "pitcher_id",
        ]
        + pitcher_feature_columns
    ].copy()

    starter = starter.rename(
        columns={
            "pitcher_id":
                "opposing_starting_pitcher_id",
            **{
                column: f"starter_{column}"
                for column
                in pitcher_feature_columns
            },
        }
    )

    interaction = interaction.merge(
        starter,
        on=[
            "game_pk",
            "opposing_starting_pitcher_id",
        ],
        how="left",
        validate="one_to_one",
    )

    interaction = _add_lineup_aggregates(
        interaction,
        batter_feature_columns,
    )

    slot_history_flags = []

    for position in range(1, 10):
        column = (
            f"slot_{position}_career_prior_games"
        )

        if column in interaction.columns:
            flag = interaction[column].notna()
        else:
            flag = pd.Series(
                False,
                index=interaction.index,
            )

        slot_history_flags.append(flag)

    history_matrix = pd.concat(
        slot_history_flags,
        axis=1,
    )

    interaction[
        "batter_snapshot_slots_matched"
    ] = history_matrix.sum(
        axis=1
    ).astype("int8")

    starter_history_column = (
        "starter_career_prior_games"
    )

    interaction[
        "starter_snapshot_matched"
    ] = (
        interaction[
            starter_history_column
        ].notna()
        if starter_history_column
        in interaction.columns
        else False
    )

    interaction[
        "complete_snapshot_join"
    ] = (
        interaction[
            "batter_snapshot_slots_matched"
        ].eq(9)
        & interaction[
            "starter_snapshot_matched"
        ]
    )

    interaction[
        "strict_backtest_safe"
    ] = True

    interaction[
        "current_game_outcomes_used"
    ] = False

    interaction[
        "same_date_games_used"
    ] = False

    interaction[
        "future_games_used"
    ] = False

    interaction[
        "prediction_or_weight_assigned"
    ] = False

    interaction[
        "interaction_engine_version"
    ] = ENGINE_VERSION

    interaction = interaction.sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return interaction


def validate_lineup_starter_inputs(
    dataframe: pd.DataFrame,
    lineups: pd.DataFrame,
    anomaly_registry: pd.DataFrame,
) -> dict[str, Any]:
    safe_game_count = int(
        anomaly_registry[
            "strict_backtest_safe"
        ].fillna(False).sum()
    )

    expected_rows = (
        safe_game_count * 2
    )

    duplicate_team_games = int(
        dataframe.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if len(dataframe) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} interaction rows; "
            f"found {len(dataframe):,}."
        )

    if duplicate_team_games:
        raise AssertionError(
            f"Found {duplicate_team_games} duplicate team-games."
        )

    unsafe_rows = int(
        (
            ~dataframe[
                "strict_backtest_safe"
            ]
        ).sum()
    )

    if unsafe_rows:
        raise AssertionError(
            f"Found {unsafe_rows} unsafe interaction rows."
        )

    full_join_rows = int(
        dataframe[
            "complete_snapshot_join"
        ].sum()
    )

    incomplete_join_rows = int(
        (
            ~dataframe[
                "complete_snapshot_join"
            ]
        ).sum()
    )

    return {
        "safe_games": safe_game_count,
        "expected_team_game_rows":
            expected_rows,
        "actual_team_game_rows":
            int(len(dataframe)),
        "duplicate_team_games":
            duplicate_team_games,
        "complete_snapshot_joins":
            full_join_rows,
        "incomplete_snapshot_joins":
            incomplete_join_rows,
        "complete_join_pct": (
            full_join_rows
            / len(dataframe)
            if len(dataframe)
            else None
        ),
        "unsafe_rows": unsafe_rows,
        "columns": int(
            len(dataframe.columns)
        ),
    }


def run_lineup_starter_input_engine() -> dict[str, Any]:
    lineups = _load_parquet(
        HISTORICAL_LINEUP_PATH,
        "historical lineups",
    )

    batter_snapshots = _load_parquet(
        BATTER_SNAPSHOT_PATH,
        "batter pregame snapshots",
    )

    pitcher_snapshots = _load_parquet(
        PITCHER_SNAPSHOT_PATH,
        "pitcher pregame snapshots",
    )

    anomaly_registry = _load_parquet(
        ANOMALY_REGISTRY_PATH,
        "anomaly registry",
    )

    interaction = (
        build_lineup_starter_inputs(
            lineups=lineups,
            batter_snapshots=batter_snapshots,
            pitcher_snapshots=pitcher_snapshots,
            anomaly_registry=anomaly_registry,
        )
    )

    validation = (
        validate_lineup_starter_inputs(
            dataframe=interaction,
            lineups=lineups,
            anomaly_registry=anomaly_registry,
        )
    )

    _atomic_parquet_write(
        interaction,
        OUTPUT_PATH,
    )

    metadata = {
        "engine": (
            "ATLAS Lineup × Starter Input Engine"
        ),
        "engine_version": ENGINE_VERSION,
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "validation": validation,
        "outputs": {
            "interaction_table": str(
                OUTPUT_PATH
            ),
        },
        "pregame_safety": {
            "current_game_outcomes_used": False,
            "same_date_games_used": False,
            "future_games_used": False,
            "quarantined_games_excluded": True,
            "predictions_created": False,
            "weights_assigned": False,
        },
    }

    _atomic_json_write(
        metadata,
        METADATA_PATH,
    )

    print("=" * 76)
    print("ATLAS LINEUP × STARTER INPUT ENGINE")
    print("=" * 76)
    print(
        f"Safe Games................ "
        f"{validation['safe_games']:,}"
    )
    print(
        f"Team-Game Rows............ "
        f"{validation['actual_team_game_rows']:,}"
    )
    print(
        f"Complete Snapshot Joins... "
        f"{validation['complete_snapshot_joins']:,}"
    )
    print(
        f"Incomplete Snapshot Joins. "
        f"{validation['incomplete_snapshot_joins']:,}"
    )
    print(
        f"Input Columns............. "
        f"{validation['columns']:,}"
    )
    print(
        f"Duplicate Team-Games...... "
        f"{validation['duplicate_team_games']:,}"
    )
    print(
        f"Unsafe Rows............... "
        f"{validation['unsafe_rows']:,}"
    )
    print(
        f"Saved To.................. "
        f"{OUTPUT_PATH}"
    )
    print("=" * 76)

    return metadata
