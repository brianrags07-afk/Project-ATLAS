
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR, MASTER_DIR


SNAPSHOT_ENGINE_VERSION = "1.1.0"

GAME_ANOMALY_REGISTRY_PATH = (
    DATA_DIR
    / "validation"
    / "anomalies"
    / "game_anomaly_registry.parquet"
)

MASTER_PITCH_PATH = (
    MASTER_DIR
    / "master_pitch_database.parquet"
)

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


HIT_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
}

WALK_EVENTS = {
    "walk",
    "intent_walk",
}

STRIKEOUT_EVENTS = {
    "strikeout",
    "strikeout_double_play",
}

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}


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


def load_regular_pitches() -> pd.DataFrame:
    if not MASTER_PITCH_PATH.exists():
        raise FileNotFoundError(
            f"Missing master pitch database: {MASTER_PITCH_PATH}"
        )

    pitches = pd.read_parquet(
        MASTER_PITCH_PATH
    )

    if "game_type" not in pitches.columns:
        raise KeyError(
            "Master pitch database is missing game_type."
        )

    game_types = set(
        pitches["game_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    if game_types != {"R"}:
        raise ValueError(
            "Snapshot engine requires regular-season-only data. "
            f"Found: {sorted(game_types)}"
        )

    if not GAME_ANOMALY_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Anomaly registry is required before building "
            f"walk-forward facts: {GAME_ANOMALY_REGISTRY_PATH}"
        )

    anomaly_registry = pd.read_parquet(
        GAME_ANOMALY_REGISTRY_PATH
    )

    unsafe_game_pks = set(
        pd.to_numeric(
            anomaly_registry.loc[
                ~anomaly_registry["strict_backtest_safe"],
                "game_pk",
            ],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    pitches = pitches[
        ~pitches["game_pk"].isin(
            unsafe_game_pks
        )
    ].copy()

    return pitches


def prepare_pitch_events(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
        "stand",
        "p_throws",
        "events",
        "description",
        "type",
        "zone",
        "pitch_type",
        "release_speed",
        "launch_speed",
        "launch_angle",
    ]

    missing = [
        column
        for column in required
        if column not in pitches.columns
    ]

    if missing:
        raise KeyError(
            f"Snapshot engine missing required columns: {missing}"
        )

    optional = [
        "release_spin_rate",
        "release_extension",
        "effective_speed",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "estimated_slg_using_speedangle",
        "bb_type",
    ]

    selected = required + [
        column
        for column in optional
        if column in pitches.columns
        and column not in required
    ]

    selected = list(dict.fromkeys(selected))

    events = pitches[selected].copy()

    events["game_date"] = pd.to_datetime(
        events["game_date"],
        errors="coerce",
    )

    top = (
        events["inning_topbot"]
        .astype("string")
        .eq("Top")
    )

    events["batting_team"] = np.where(
        top,
        events["away_team"],
        events["home_team"],
    )

    events["pitching_team"] = np.where(
        top,
        events["home_team"],
        events["away_team"],
    )

    events["batter_home_away"] = np.where(
        top,
        "AWAY",
        "HOME",
    )

    events["pitcher_home_away"] = np.where(
        top,
        "HOME",
        "AWAY",
    )

    description = events["description"].astype("string")
    event = events["events"].astype("string")
    pitch_result = events["type"].astype("string")

    events["_is_pa_end"] = events["events"].notna()
    events["_is_hit"] = event.isin(HIT_EVENTS)
    events["_is_home_run"] = event.eq("home_run")
    events["_is_walk"] = event.isin(WALK_EVENTS)
    events["_is_strikeout"] = event.isin(STRIKEOUT_EVENTS)

    events["_is_swing"] = description.isin(
        SWING_DESCRIPTIONS
    )
    events["_is_whiff"] = description.isin(
        WHIFF_DESCRIPTIONS
    )
    events["_is_called_strike"] = description.eq(
        "called_strike"
    )

    events["_is_strike"] = (
        pitch_result.eq("S")
        | pitch_result.eq("X")
    )
    events["_is_ball"] = pitch_result.eq("B")
    events["_is_in_play"] = pitch_result.eq("X")

    events["_is_batted_ball"] = (
        events["_is_in_play"]
        & events["launch_speed"].notna()
    )

    events["_is_hard_hit"] = (
        pd.to_numeric(
            events["launch_speed"],
            errors="coerce",
        ).ge(95)
        & events["_is_batted_ball"]
    )

    zone = pd.to_numeric(
        events["zone"],
        errors="coerce",
    )

    events["_is_zone"] = zone.between(1, 9)
    events["_is_out_zone"] = zone.between(11, 14)

    events["_is_chase"] = (
        events["_is_swing"]
        & events["_is_out_zone"]
    )

    if {
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
    }.issubset(events.columns):
        plate_x = pd.to_numeric(
            events["plate_x"],
            errors="coerce",
        )

        plate_z = pd.to_numeric(
            events["plate_z"],
            errors="coerce",
        )

        sz_top = pd.to_numeric(
            events["sz_top"],
            errors="coerce",
        )

        sz_bot = pd.to_numeric(
            events["sz_bot"],
            errors="coerce",
        )

        zone_height = (
            sz_top - sz_bot
        ).replace(0, np.nan)

        normalized_z = (
            plate_z - sz_bot
        ) / zone_height

        events["_is_heart"] = (
            plate_x.abs().le(0.55)
            & normalized_z.between(
                0.28,
                0.72,
            )
        )

        events["_is_middle_middle"] = (
            plate_x.abs().le(0.30)
            & normalized_z.between(
                0.38,
                0.62,
            )
        )
    else:
        events["_is_heart"] = False
        events["_is_middle_middle"] = False

    return events


def _sum_flags(
    grouped: Any,
    column: str,
) -> pd.Series:
    return grouped[column].sum().astype("int64")


def build_batter_game_facts(
    events: pd.DataFrame,
) -> pd.DataFrame:
    batter_events = events.dropna(
        subset=["batter"]
    ).copy()

    batter_events["player_id"] = (
        batter_events["batter"]
        .astype("int64")
    )

    keys = [
        "game_pk",
        "game_date",
        "atlas_season",
        "player_id",
        "batting_team",
        "pitching_team",
        "batter_home_away",
    ]

    grouped = batter_events.groupby(
        keys,
        sort=True,
        dropna=False,
    )

    facts = grouped.size().rename(
        "pitches_seen"
    ).reset_index()

    flag_columns = {
        "plate_appearances": "_is_pa_end",
        "hits": "_is_hit",
        "home_runs": "_is_home_run",
        "walks": "_is_walk",
        "strikeouts": "_is_strikeout",
        "swings": "_is_swing",
        "whiffs": "_is_whiff",
        "called_strikes": "_is_called_strike",
        "balls_seen": "_is_ball",
        "out_zone_pitches": "_is_out_zone",
        "chase_swings": "_is_chase",
        "batted_balls": "_is_batted_ball",
        "hard_hit_balls": "_is_hard_hit",
    }

    for output, source in flag_columns.items():
        values = (
            grouped[source]
            .sum()
            .rename(output)
            .reset_index()
        )

        facts = facts.merge(
            values,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    contact = (
        batter_events[
            batter_events["_is_batted_ball"]
        ]
        .groupby(
            keys,
            sort=True,
            dropna=False,
        )
        .agg(
            exit_velocity_sum=(
                "launch_speed",
                "sum",
            ),
            launch_angle_sum=(
                "launch_angle",
                "sum",
            ),
        )
        .reset_index()
    )

    facts = facts.merge(
        contact,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    numeric_fill = [
        "exit_velocity_sum",
        "launch_angle_sum",
    ]

    facts[numeric_fill] = (
        facts[numeric_fill]
        .fillna(0.0)
    )

    facts["snapshot_engine_version"] = (
        SNAPSHOT_ENGINE_VERSION
    )

    return facts


def build_pitcher_game_facts(
    events: pd.DataFrame,
) -> pd.DataFrame:
    pitcher_events = events.dropna(
        subset=["pitcher"]
    ).copy()

    pitcher_events["pitcher_id"] = (
        pitcher_events["pitcher"]
        .astype("int64")
    )

    keys = [
        "game_pk",
        "game_date",
        "atlas_season",
        "pitcher_id",
        "pitching_team",
        "batting_team",
        "pitcher_home_away",
    ]

    grouped = pitcher_events.groupby(
        keys,
        sort=True,
        dropna=False,
    )

    facts = grouped.size().rename(
        "pitches_thrown"
    ).reset_index()

    flag_columns = {
        "plate_appearances": "_is_pa_end",
        "strikeouts": "_is_strikeout",
        "walks": "_is_walk",
        "hits_allowed": "_is_hit",
        "home_runs_allowed": "_is_home_run",
        "strikes_thrown": "_is_strike",
        "balls_thrown": "_is_ball",
        "swings": "_is_swing",
        "whiffs": "_is_whiff",
        "called_strikes": "_is_called_strike",
        "zone_pitches": "_is_zone",
        "out_zone_pitches": "_is_out_zone",
        "chase_swings": "_is_chase",
        "heart_pitches": "_is_heart",
        "middle_middle_pitches": "_is_middle_middle",
        "batted_balls": "_is_batted_ball",
        "hard_hit_balls": "_is_hard_hit",
    }

    for output, source in flag_columns.items():
        values = (
            grouped[source]
            .sum()
            .rename(output)
            .reset_index()
        )

        facts = facts.merge(
            values,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    summed_metrics = {
        "velocity_sum": "release_speed",
        "exit_velocity_sum": "launch_speed",
        "launch_angle_sum": "launch_angle",
    }

    for output, source in summed_metrics.items():
        values = (
            pitcher_events.assign(
                **{
                    source: pd.to_numeric(
                        pitcher_events[source],
                        errors="coerce",
                    )
                }
            )
            .groupby(
                keys,
                sort=True,
                dropna=False,
            )[source]
            .sum(min_count=1)
            .rename(output)
            .reset_index()
        )

        facts = facts.merge(
            values,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    facts[
        [
            "velocity_sum",
            "exit_velocity_sum",
            "launch_angle_sum",
        ]
    ] = facts[
        [
            "velocity_sum",
            "exit_velocity_sum",
            "launch_angle_sum",
        ]
    ].fillna(0.0)

    facts["snapshot_engine_version"] = (
        SNAPSHOT_ENGINE_VERSION
    )

    return facts


def run_game_fact_builder() -> dict[str, Any]:
    pitches = load_regular_pitches()
    events = prepare_pitch_events(pitches)

    batter_facts = build_batter_game_facts(
        events
    )

    pitcher_facts = build_pitcher_game_facts(
        events
    )

    _atomic_parquet_write(
        batter_facts,
        BATTER_GAME_FACTS_PATH,
    )

    _atomic_parquet_write(
        pitcher_facts,
        PITCHER_GAME_FACTS_PATH,
    )

    summary = {
        "engine": "ATLAS Walk-Forward Snapshot Engine",
        "engine_version": SNAPSHOT_ENGINE_VERSION,
        "source_pitch_rows": int(
            len(pitches)
        ),
        "batter_game_rows": int(
            len(batter_facts)
        ),
        "unique_batters": int(
            batter_facts[
                "player_id"
            ].nunique()
        ),
        "pitcher_game_rows": int(
            len(pitcher_facts)
        ),
        "unique_pitchers": int(
            pitcher_facts[
                "pitcher_id"
            ].nunique()
        ),
        "batter_output": str(
            BATTER_GAME_FACTS_PATH
        ),
        "pitcher_output": str(
            PITCHER_GAME_FACTS_PATH
        ),
        "pregame_safety": {
            "these_tables_contain_single_game_facts": True,
            "not_safe_until_shifted_before_join": True,
            "future_games_used_in_each_row": False,
            "predictions_created": False,
        },
    }

    print("=" * 72)
    print("ATLAS WALK-FORWARD GAME FACT BUILDER")
    print("=" * 72)
    print(
        f"Source Pitch Rows........ "
        f"{summary['source_pitch_rows']:,}"
    )
    print(
        f"Batter-Game Rows......... "
        f"{summary['batter_game_rows']:,}"
    )
    print(
        f"Unique Batters........... "
        f"{summary['unique_batters']:,}"
    )
    print(
        f"Pitcher-Game Rows........ "
        f"{summary['pitcher_game_rows']:,}"
    )
    print(
        f"Unique Pitchers.......... "
        f"{summary['unique_pitchers']:,}"
    )
    print("=" * 72)

    return summary
