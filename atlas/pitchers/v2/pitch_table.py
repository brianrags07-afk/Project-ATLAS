
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import MASTER_DIR
from atlas.pitchers.v2.definitions import (
    FOUL_DESCRIPTIONS,
    HIT_EVENTS,
    PITCH_COUNT_BINS,
    PITCH_COUNT_LABELS,
    STRIKEOUT_EVENTS,
    SWING_DESCRIPTIONS,
    WALK_EVENTS,
    WHIFF_DESCRIPTIONS,
)


MASTER_PITCH_PATH = (
    MASTER_DIR
    / "master_pitch_database.parquet"
)


def load_master_pitches() -> pd.DataFrame:
    if not MASTER_PITCH_PATH.exists():
        raise FileNotFoundError(
            f"Missing master pitch database: {MASTER_PITCH_PATH}"
        )

    pitches = pd.read_parquet(MASTER_PITCH_PATH)

    if "game_type" not in pitches.columns:
        raise KeyError(
            "master_pitch_database.parquet is missing game_type"
        )

    types = set(
        pitches["game_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    if types != {"R"}:
        raise ValueError(
            "Pitcher Engine v2 requires the active regular-season "
            f"master table. Found game types: {sorted(types)}"
        )

    return pitches


def _series(
    df: pd.DataFrame,
    column: str,
    default=pd.NA,
) -> pd.Series:
    if column in df.columns:
        return df[column]

    return pd.Series(
        default,
        index=df.index,
    )


def build_pitcher_pitch_table(
    pitches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pitches is None:
        pitches = load_master_pitches()

    required = [
        "game_pk",
        "game_date",
        "atlas_season",
        "game_type",
        "pitcher",
        "p_throws",
        "batter",
        "stand",
        "home_team",
        "away_team",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "balls",
        "strikes",
        "outs_when_up",
        "pitch_type",
        "description",
        "events",
        "type",
        "zone",
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
            f"Pitcher Engine v2 missing required columns: {missing}"
        )

    optional = [
        "pitch_name",
        "release_spin_rate",
        "release_extension",
        "effective_speed",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "estimated_slg_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "bb_type",
        "launch_speed_angle",
        "hc_x",
        "hc_y",
        "on_1b",
        "on_2b",
        "on_3b",
        "bat_score",
        "fld_score",
        "post_bat_score",
        "post_fld_score",
        "delta_home_win_exp",
        "delta_run_exp",
        "if_fielding_alignment",
        "of_fielding_alignment",
        "umpire",
        "sv_id",
    ]

    selected = required + [
        column
        for column in optional
        if column in pitches.columns
        and column not in required
    ]

    selected = list(dict.fromkeys(selected))
    table = pitches[selected].copy()

    table = table.rename(
        columns={
            "pitcher": "pitcher_id",
            "p_throws": "throws",
            "batter": "batter_id",
            "stand": "batter_side",
        }
    )

    table = table.dropna(
        subset=["pitcher_id"]
    )

    table["pitcher_id"] = (
        table["pitcher_id"]
        .astype("int64")
    )

    table["game_pk"] = (
        table["game_pk"]
        .astype("int64")
    )

    table["game_date"] = pd.to_datetime(
        table["game_date"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Team context
    # --------------------------------------------------------

    top = (
        table["inning_topbot"]
        .astype("string")
        .eq("Top")
    )

    table["team"] = np.where(
        top,
        table["home_team"],
        table["away_team"],
    )

    table["opponent"] = np.where(
        top,
        table["away_team"],
        table["home_team"],
    )

    table["home_away"] = np.where(
        top,
        "HOME",
        "AWAY",
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    table["_half_order"] = (
        table["inning_topbot"]
        .astype("string")
        .map({"Top": 0, "Bot": 1})
        .fillna(2)
        .astype("int8")
    )

    table = table.sort_values(
        [
            "game_pk",
            "inning",
            "_half_order",
            "at_bat_number",
            "pitch_number",
        ],
        kind="stable",
    ).reset_index(drop=True)

    table["appearance_pitch_count"] = (
        table.groupby(
            ["game_pk", "pitcher_id"]
        )
        .cumcount()
        .add(1)
        .astype("int32")
    )

    table["pitch_count_bucket"] = pd.cut(
        table["appearance_pitch_count"],
        bins=PITCH_COUNT_BINS,
        labels=PITCH_COUNT_LABELS,
        include_lowest=True,
    ).astype("string")

    # --------------------------------------------------------
    # Starter/reliever role
    # --------------------------------------------------------

    first_pitcher = (
        table.groupby(
            ["game_pk", "team"],
            sort=False,
        )["pitcher_id"]
        .first()
        .rename("_first_pitcher_id")
        .reset_index()
    )

    table = table.merge(
        first_pitcher,
        on=["game_pk", "team"],
        how="left",
        validate="many_to_one",
    )

    table["role"] = np.where(
        table["pitcher_id"]
        .eq(table["_first_pitcher_id"]),
        "starter_or_opener",
        "reliever",
    )

    # --------------------------------------------------------
    # Count state
    # --------------------------------------------------------

    balls = pd.to_numeric(
        table["balls"],
        errors="coerce",
    )

    strikes = pd.to_numeric(
        table["strikes"],
        errors="coerce",
    )

    table["exact_count"] = (
        balls.fillna(-1)
        .astype(int)
        .astype(str)
        + "-"
        + strikes.fillna(-1)
        .astype(int)
        .astype(str)
    )

    table["count_state"] = np.select(
        [
            strikes.eq(2),
            balls.eq(3),
            balls.gt(strikes),
            strikes.gt(balls),
        ],
        [
            "two_strike",
            "three_ball",
            "behind",
            "ahead",
        ],
        default="even",
    )

    # --------------------------------------------------------
    # Times facing each batter
    # --------------------------------------------------------

    pa_keys = (
        table[
            [
                "game_pk",
                "pitcher_id",
                "batter_id",
                "at_bat_number",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "game_pk",
                "pitcher_id",
                "batter_id",
                "at_bat_number",
            ],
            kind="stable",
        )
    )

    pa_keys["batter_times_faced"] = (
        pa_keys.groupby(
            [
                "game_pk",
                "pitcher_id",
                "batter_id",
            ]
        )
        .cumcount()
        .add(1)
        .astype("int16")
    )

    table = table.merge(
        pa_keys,
        on=[
            "game_pk",
            "pitcher_id",
            "batter_id",
            "at_bat_number",
        ],
        how="left",
        validate="many_to_one",
    )

    table["times_through_order"] = np.select(
        [
            table["batter_times_faced"].eq(1),
            table["batter_times_faced"].eq(2),
            table["batter_times_faced"].ge(3),
        ],
        [
            "first",
            "second",
            "third_plus",
        ],
        default="unknown",
    )

    # --------------------------------------------------------
    # Base state and score state
    # --------------------------------------------------------

    for column in [
        "on_1b",
        "on_2b",
        "on_3b",
    ]:
        if column not in table.columns:
            table[column] = pd.NA

    table["runners_on"] = (
        table[
            ["on_1b", "on_2b", "on_3b"]
        ]
        .notna()
        .sum(axis=1)
        .astype("int8")
    )

    table["base_state"] = np.select(
        [
            table["runners_on"].eq(0),
            table["on_2b"].notna()
            | table["on_3b"].notna(),
        ],
        [
            "bases_empty",
            "risp",
        ],
        default="runner_on_first_only",
    )

    bat_score = pd.to_numeric(
        _series(table, "bat_score"),
        errors="coerce",
    )

    fld_score = pd.to_numeric(
        _series(table, "fld_score"),
        errors="coerce",
    )

    table["pitcher_team_lead"] = (
        fld_score - bat_score
    )

    table["score_state"] = np.select(
        [
            table["pitcher_team_lead"].gt(0),
            table["pitcher_team_lead"].lt(0),
        ],
        [
            "leading",
            "trailing",
        ],
        default="tied",
    )

    # --------------------------------------------------------
    # Pitch outcomes
    # --------------------------------------------------------

    description = (
        table["description"]
        .astype("string")
    )

    event = (
        table["events"]
        .astype("string")
    )

    pitch_result = (
        table["type"]
        .astype("string")
    )

    table["_is_strike"] = (
        pitch_result.eq("S")
        | pitch_result.eq("X")
    )

    table["_is_ball"] = (
        pitch_result.eq("B")
    )

    table["_is_swing"] = (
        description.isin(
            SWING_DESCRIPTIONS
        )
    )

    table["_is_whiff"] = (
        description.isin(
            WHIFF_DESCRIPTIONS
        )
    )

    table["_is_called_strike"] = (
        description.eq(
            "called_strike"
        )
    )

    table["_is_foul"] = (
        description.isin(
            FOUL_DESCRIPTIONS
        )
    )

    table["_is_in_play"] = (
        pitch_result.eq("X")
    )

    table["_is_pa_end"] = (
        table["events"].notna()
    )

    table["_is_strikeout"] = (
        event.isin(
            STRIKEOUT_EVENTS
        )
    )

    table["_is_walk"] = (
        event.isin(
            WALK_EVENTS
        )
    )

    table["_is_hit"] = (
        event.isin(
            HIT_EVENTS
        )
    )

    table["_is_home_run"] = (
        event.eq("home_run")
    )

    table["_is_first_pitch"] = (
        pd.to_numeric(
            table["pitch_number"],
            errors="coerce",
        ).eq(1)
    )

    table["_is_two_strike_pitch"] = (
        strikes.eq(2)
    )

    table["_is_three_ball_pitch"] = (
        balls.eq(3)
    )

    table["_is_full_count"] = (
        balls.eq(3)
        & strikes.eq(2)
    )

    # --------------------------------------------------------
    # Zone / location / danger
    # --------------------------------------------------------

    zone = pd.to_numeric(
        table["zone"],
        errors="coerce",
    )

    table["_is_zone"] = (
        zone.between(1, 9)
    )

    table["_is_out_zone"] = (
        zone.between(11, 14)
    )

    table["_is_chase"] = (
        table["_is_swing"]
        & table["_is_out_zone"]
    )

    plate_x = pd.to_numeric(
        _series(table, "plate_x"),
        errors="coerce",
    )

    plate_z = pd.to_numeric(
        _series(table, "plate_z"),
        errors="coerce",
    )

    sz_top = pd.to_numeric(
        _series(table, "sz_top"),
        errors="coerce",
    )

    sz_bot = pd.to_numeric(
        _series(table, "sz_bot"),
        errors="coerce",
    )

    zone_height = sz_top - sz_bot

    normalized_z = (
        (plate_z - sz_bot)
        / zone_height.replace(0, np.nan)
    )

    table["_is_heart"] = (
        plate_x.abs().le(0.55)
        & normalized_z.between(
            0.28,
            0.72,
        )
    )

    table["_is_middle_middle"] = (
        plate_x.abs().le(0.30)
        & normalized_z.between(
            0.38,
            0.62,
        )
    )

    table["_is_shadow"] = (
        (
            plate_x.abs().between(
                0.55,
                1.10,
                inclusive="both",
            )
            & normalized_z.between(
                0.05,
                0.95,
            )
        )
        | (
            plate_x.abs().le(1.10)
            & (
                normalized_z.between(
                    0.05,
                    0.28,
                )
                | normalized_z.between(
                    0.72,
                    0.95,
                )
            )
        )
    )

    table["_is_waste"] = (
        plate_x.abs().gt(1.50)
        | normalized_z.lt(-0.10)
        | normalized_z.gt(1.10)
    )

    table["_is_edge"] = (
        table["_is_shadow"]
        & ~table["_is_waste"]
    )

    table["_is_two_strike_heart"] = (
        table["_is_two_strike_pitch"]
        & table["_is_heart"]
    )

    table["_is_three_ball_heart"] = (
        table["_is_three_ball_pitch"]
        & table["_is_heart"]
    )

    # --------------------------------------------------------
    # Contact outcomes
    # --------------------------------------------------------

    table["_is_batted_ball"] = (
        table["_is_in_play"]
        & table["launch_speed"].notna()
    )

    table["_is_hard_hit"] = (
        pd.to_numeric(
            table["launch_speed"],
            errors="coerce",
        ).ge(95)
        & table["_is_batted_ball"]
    )

    bb_type = (
        _series(table, "bb_type")
        .astype("string")
    )

    table["_is_ground_ball"] = (
        bb_type.eq("ground_ball")
    )

    table["_is_fly_ball"] = (
        bb_type.eq("fly_ball")
    )

    table["_is_line_drive"] = (
        bb_type.eq("line_drive")
    )

    table["_is_popup"] = (
        bb_type.eq("popup")
    )

    # --------------------------------------------------------
    # Event identity and sequencing
    # --------------------------------------------------------

    table["pitch_event_key"] = (
        table["game_pk"]
        .astype(str)
        + "_"
        + table["at_bat_number"]
        .astype(str)
        + "_"
        + table["pitch_number"]
        .astype(str)
    )

    table["previous_pitch_type"] = (
        table.groupby(
            ["game_pk", "pitcher_id"]
        )["pitch_type"]
        .shift(1)
    )

    table["previous_pitch_result"] = (
        table.groupby(
            ["game_pk", "pitcher_id"]
        )["description"]
        .shift(1)
    )

    table["velocity_change_from_previous"] = (
        pd.to_numeric(
            table["release_speed"],
            errors="coerce",
        )
        - pd.to_numeric(
            table.groupby(
                ["game_pk", "pitcher_id"]
            )["release_speed"]
            .shift(1),
            errors="coerce",
        )
    )

    table = table.drop(
        columns=[
            "_half_order",
            "_first_pitcher_id",
        ],
        errors="ignore",
    )

    return table
