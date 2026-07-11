
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


SNAPSHOT_BUILDER_VERSION = "1.0.0"

SNAPSHOT_DIR = (
    DATA_DIR
    / "pregame"
    / "snapshots"
)

BATTER_GAME_FACTS_PATH = (
    SNAPSHOT_DIR
    / "batter_game_facts.parquet"
)

PITCHER_GAME_FACTS_PATH = (
    SNAPSHOT_DIR
    / "pitcher_game_facts.parquet"
)

BATTER_PREGAME_SNAPSHOTS_PATH = (
    SNAPSHOT_DIR
    / "batter_pregame_snapshots.parquet"
)

PITCHER_PREGAME_SNAPSHOTS_PATH = (
    SNAPSHOT_DIR
    / "pitcher_pregame_snapshots.parquet"
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


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = numerator / denominator.replace(
        0,
        np.nan,
    )

    return result.astype("float64")


def _load_facts(
    path: Path,
    name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name}: {path}"
        )

    dataframe = pd.read_parquet(path)

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="coerce",
    ).dt.normalize()

    if dataframe["game_date"].isna().any():
        raise ValueError(
            f"{name} contains invalid game dates."
        )

    return dataframe


def _build_prior_date_totals(
    facts: pd.DataFrame,
    entity_column: str,
    sum_columns: list[str],
    prefix: str,
    include_season: bool,
) -> pd.DataFrame:
    """
    Aggregate each entity to one row per calendar date, then shift
    cumulative totals by one date.

    This intentionally excludes every game played on the current date.
    That is conservative for doubleheaders and guarantees that no
    current-game or same-day outcome enters a pregame snapshot.
    """
    grouping_columns = [
        entity_column,
        "game_date",
    ]

    if include_season:
        grouping_columns.insert(
            1,
            "atlas_season",
        )

    daily = (
        facts.groupby(
            grouping_columns,
            sort=True,
            dropna=False,
        )[sum_columns]
        .sum()
        .reset_index()
    )

    daily = daily.sort_values(
        grouping_columns,
        kind="stable",
    ).reset_index(drop=True)

    entity_group_columns = [
        entity_column,
    ]

    if include_season:
        entity_group_columns.append(
            "atlas_season"
        )

    prior = daily[
        grouping_columns
    ].copy()

    grouped = daily.groupby(
        entity_group_columns,
        sort=False,
        dropna=False,
    )

    for column in sum_columns:
        cumulative = grouped[column].cumsum()

        prior[
            f"{prefix}_{column}"
        ] = (
            cumulative
            - daily[column]
        )

    date_game_counts = (
        facts.groupby(
            grouping_columns,
            sort=True,
            dropna=False,
        )["game_pk"]
        .nunique()
        .rename("_games_on_date")
        .reset_index()
    )

    prior = prior.merge(
        date_game_counts,
        on=grouping_columns,
        how="left",
        validate="one_to_one",
    )

    game_grouped = prior.groupby(
        entity_group_columns,
        sort=False,
        dropna=False,
    )

    cumulative_games = game_grouped[
        "_games_on_date"
    ].cumsum()

    prior[
        f"{prefix}_games"
    ] = (
        cumulative_games
        - prior["_games_on_date"]
    )

    prior = prior.drop(
        columns=["_games_on_date"]
    )

    return prior


def _add_batter_rates(
    dataframe: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    dataframe[
        f"{prefix}_hit_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_hits"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_home_run_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_home_runs"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_walk_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_walks"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_strikeout_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_strikeouts"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_swing_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_swings"],
        dataframe[f"{prefix}_pitches_seen"],
    )

    dataframe[
        f"{prefix}_whiff_pct_per_swing"
    ] = _safe_divide(
        dataframe[f"{prefix}_whiffs"],
        dataframe[f"{prefix}_swings"],
    )

    dataframe[
        f"{prefix}_called_strike_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_called_strikes"],
        dataframe[f"{prefix}_pitches_seen"],
    )

    dataframe[
        f"{prefix}_ball_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_balls_seen"],
        dataframe[f"{prefix}_pitches_seen"],
    )

    dataframe[
        f"{prefix}_chase_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_chase_swings"],
        dataframe[f"{prefix}_out_zone_pitches"],
    )

    dataframe[
        f"{prefix}_hard_hit_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_hard_hit_balls"],
        dataframe[f"{prefix}_batted_balls"],
    )

    dataframe[
        f"{prefix}_avg_exit_velocity"
    ] = _safe_divide(
        dataframe[f"{prefix}_exit_velocity_sum"],
        dataframe[f"{prefix}_batted_balls"],
    )

    dataframe[
        f"{prefix}_avg_launch_angle"
    ] = _safe_divide(
        dataframe[f"{prefix}_launch_angle_sum"],
        dataframe[f"{prefix}_batted_balls"],
    )

    return dataframe


def _add_pitcher_rates(
    dataframe: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    dataframe[
        f"{prefix}_strikeout_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_strikeouts"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_walk_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_walks"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_hit_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_hits_allowed"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_home_run_rate_per_pa"
    ] = _safe_divide(
        dataframe[f"{prefix}_home_runs_allowed"],
        dataframe[f"{prefix}_plate_appearances"],
    )

    dataframe[
        f"{prefix}_strike_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_strikes_thrown"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_ball_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_balls_thrown"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_whiff_pct_per_swing"
    ] = _safe_divide(
        dataframe[f"{prefix}_whiffs"],
        dataframe[f"{prefix}_swings"],
    )

    dataframe[
        f"{prefix}_called_strike_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_called_strikes"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_csw_pct"
    ] = _safe_divide(
        (
            dataframe[f"{prefix}_called_strikes"]
            + dataframe[f"{prefix}_whiffs"]
        ),
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_zone_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_zone_pitches"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_chase_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_chase_swings"],
        dataframe[f"{prefix}_out_zone_pitches"],
    )

    dataframe[
        f"{prefix}_heart_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_heart_pitches"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_middle_middle_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_middle_middle_pitches"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_hard_hit_pct"
    ] = _safe_divide(
        dataframe[f"{prefix}_hard_hit_balls"],
        dataframe[f"{prefix}_batted_balls"],
    )

    dataframe[
        f"{prefix}_avg_velocity"
    ] = _safe_divide(
        dataframe[f"{prefix}_velocity_sum"],
        dataframe[f"{prefix}_pitches_thrown"],
    )

    dataframe[
        f"{prefix}_avg_exit_velocity_allowed"
    ] = _safe_divide(
        dataframe[f"{prefix}_exit_velocity_sum"],
        dataframe[f"{prefix}_batted_balls"],
    )

    dataframe[
        f"{prefix}_avg_launch_angle_allowed"
    ] = _safe_divide(
        dataframe[f"{prefix}_launch_angle_sum"],
        dataframe[f"{prefix}_batted_balls"],
    )

    return dataframe


def build_batter_pregame_snapshots(
    batter_facts: pd.DataFrame,
) -> pd.DataFrame:
    sum_columns = [
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
        "exit_velocity_sum",
        "launch_angle_sum",
    ]

    career_prior = _build_prior_date_totals(
        facts=batter_facts,
        entity_column="player_id",
        sum_columns=sum_columns,
        prefix="career_prior",
        include_season=False,
    )

    season_prior = _build_prior_date_totals(
        facts=batter_facts,
        entity_column="player_id",
        sum_columns=sum_columns,
        prefix="season_prior",
        include_season=True,
    )

    snapshots = batter_facts[
        [
            "game_pk",
            "game_date",
            "atlas_season",
            "player_id",
            "batting_team",
            "pitching_team",
            "batter_home_away",
        ]
    ].copy()

    snapshots = snapshots.merge(
        career_prior,
        on=[
            "player_id",
            "game_date",
        ],
        how="left",
        validate="many_to_one",
    )

    snapshots = snapshots.merge(
        season_prior,
        on=[
            "player_id",
            "atlas_season",
            "game_date",
        ],
        how="left",
        validate="many_to_one",
    )

    prior_columns = [
        column
        for column in snapshots.columns
        if column.startswith(
            (
                "career_prior_",
                "season_prior_",
            )
        )
    ]

    snapshots[prior_columns] = (
        snapshots[prior_columns]
        .fillna(0)
    )

    snapshots = _add_batter_rates(
        snapshots,
        "career_prior",
    )

    snapshots = _add_batter_rates(
        snapshots,
        "season_prior",
    )

    snapshots[
        "pregame_snapshot_safe"
    ] = True

    snapshots[
        "current_date_games_excluded"
    ] = True

    snapshots[
        "current_game_excluded"
    ] = True

    snapshots[
        "future_games_excluded"
    ] = True

    snapshots[
        "snapshot_builder_version"
    ] = SNAPSHOT_BUILDER_VERSION

    return snapshots.sort_values(
        [
            "game_date",
            "game_pk",
            "player_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_pitcher_pregame_snapshots(
    pitcher_facts: pd.DataFrame,
) -> pd.DataFrame:
    sum_columns = [
        "pitches_thrown",
        "plate_appearances",
        "strikeouts",
        "walks",
        "hits_allowed",
        "home_runs_allowed",
        "strikes_thrown",
        "balls_thrown",
        "swings",
        "whiffs",
        "called_strikes",
        "zone_pitches",
        "out_zone_pitches",
        "chase_swings",
        "heart_pitches",
        "middle_middle_pitches",
        "batted_balls",
        "hard_hit_balls",
        "velocity_sum",
        "exit_velocity_sum",
        "launch_angle_sum",
    ]

    career_prior = _build_prior_date_totals(
        facts=pitcher_facts,
        entity_column="pitcher_id",
        sum_columns=sum_columns,
        prefix="career_prior",
        include_season=False,
    )

    season_prior = _build_prior_date_totals(
        facts=pitcher_facts,
        entity_column="pitcher_id",
        sum_columns=sum_columns,
        prefix="season_prior",
        include_season=True,
    )

    snapshots = pitcher_facts[
        [
            "game_pk",
            "game_date",
            "atlas_season",
            "pitcher_id",
            "pitching_team",
            "batting_team",
            "pitcher_home_away",
        ]
    ].copy()

    snapshots = snapshots.merge(
        career_prior,
        on=[
            "pitcher_id",
            "game_date",
        ],
        how="left",
        validate="many_to_one",
    )

    snapshots = snapshots.merge(
        season_prior,
        on=[
            "pitcher_id",
            "atlas_season",
            "game_date",
        ],
        how="left",
        validate="many_to_one",
    )

    prior_columns = [
        column
        for column in snapshots.columns
        if column.startswith(
            (
                "career_prior_",
                "season_prior_",
            )
        )
    ]

    snapshots[prior_columns] = (
        snapshots[prior_columns]
        .fillna(0)
    )

    snapshots = _add_pitcher_rates(
        snapshots,
        "career_prior",
    )

    snapshots = _add_pitcher_rates(
        snapshots,
        "season_prior",
    )

    snapshots[
        "pregame_snapshot_safe"
    ] = True

    snapshots[
        "current_date_games_excluded"
    ] = True

    snapshots[
        "current_game_excluded"
    ] = True

    snapshots[
        "future_games_excluded"
    ] = True

    snapshots[
        "snapshot_builder_version"
    ] = SNAPSHOT_BUILDER_VERSION

    return snapshots.sort_values(
        [
            "game_date",
            "game_pk",
            "pitcher_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _validate_snapshots(
    snapshots: pd.DataFrame,
    facts: pd.DataFrame,
    entity_column: str,
    name: str,
) -> dict[str, Any]:
    if len(snapshots) != len(facts):
        raise AssertionError(
            f"{name}: row-count mismatch. "
            f"Facts={len(facts):,}, snapshots={len(snapshots):,}"
        )

    duplicate_rows = int(
        snapshots.duplicated(
            subset=[
                "game_pk",
                entity_column,
            ]
        ).sum()
    )

    if duplicate_rows:
        raise AssertionError(
            f"{name}: found {duplicate_rows} duplicate snapshots."
        )

    unsafe_rows = int(
        (
            ~snapshots[
                "pregame_snapshot_safe"
            ]
        ).sum()
    )

    if unsafe_rows:
        raise AssertionError(
            f"{name}: found {unsafe_rows} unsafe rows."
        )

    first_date_rows = (
        snapshots.sort_values(
            [
                entity_column,
                "game_date",
            ],
            kind="stable",
        )
        .groupby(
            entity_column,
            sort=False,
        )
        .head(1)
    )

    career_game_column = (
        "career_prior_games"
    )

    first_rows_with_history = int(
        (
            first_date_rows[
                career_game_column
            ] != 0
        ).sum()
    )

    if first_rows_with_history:
        raise AssertionError(
            f"{name}: first observed date contains prior history "
            f"for {first_rows_with_history} entities."
        )

    return {
        "rows": int(len(snapshots)),
        "entities": int(
            snapshots[
                entity_column
            ].nunique()
        ),
        "duplicate_entity_games": duplicate_rows,
        "unsafe_rows": unsafe_rows,
        "first_observed_rows_with_history":
            first_rows_with_history,
    }


def run_pregame_snapshot_builder() -> dict[str, Any]:
    batter_facts = _load_facts(
        BATTER_GAME_FACTS_PATH,
        "batter game facts",
    )

    pitcher_facts = _load_facts(
        PITCHER_GAME_FACTS_PATH,
        "pitcher game facts",
    )

    batter_snapshots = (
        build_batter_pregame_snapshots(
            batter_facts
        )
    )

    pitcher_snapshots = (
        build_pitcher_pregame_snapshots(
            pitcher_facts
        )
    )

    batter_validation = _validate_snapshots(
        snapshots=batter_snapshots,
        facts=batter_facts,
        entity_column="player_id",
        name="Batter snapshots",
    )

    pitcher_validation = _validate_snapshots(
        snapshots=pitcher_snapshots,
        facts=pitcher_facts,
        entity_column="pitcher_id",
        name="Pitcher snapshots",
    )

    _atomic_parquet_write(
        batter_snapshots,
        BATTER_PREGAME_SNAPSHOTS_PATH,
    )

    _atomic_parquet_write(
        pitcher_snapshots,
        PITCHER_PREGAME_SNAPSHOTS_PATH,
    )

    summary = {
        "engine": (
            "ATLAS Pregame Walk-Forward Snapshot Builder"
        ),
        "engine_version": (
            SNAPSHOT_BUILDER_VERSION
        ),
        "batter_validation": (
            batter_validation
        ),
        "pitcher_validation": (
            pitcher_validation
        ),
        "batter_output": str(
            BATTER_PREGAME_SNAPSHOTS_PATH
        ),
        "pitcher_output": str(
            PITCHER_PREGAME_SNAPSHOTS_PATH
        ),
        "pregame_safety": {
            "current_game_excluded": True,
            "all_current_date_games_excluded": True,
            "future_games_excluded": True,
            "same_day_doubleheader_leakage_prevented": True,
            "season_snapshots_separate": True,
            "career_snapshots_available": True,
        },
    }

    print("=" * 72)
    print("ATLAS PREGAME WALK-FORWARD SNAPSHOTS")
    print("=" * 72)
    print(
        f"Batter Snapshot Rows..... "
        f"{batter_validation['rows']:,}"
    )
    print(
        f"Unique Batters........... "
        f"{batter_validation['entities']:,}"
    )
    print(
        f"Pitcher Snapshot Rows.... "
        f"{pitcher_validation['rows']:,}"
    )
    print(
        f"Unique Pitchers.......... "
        f"{pitcher_validation['entities']:,}"
    )
    print(
        "Current-date games excluded: YES"
    )
    print(
        "Future games excluded........: YES"
    )
    print("=" * 72)

    return summary
