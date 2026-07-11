
from __future__ import annotations

from typing import Any

import pandas as pd


def safe_rate(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return None

    return float(numerator) / float(denominator)


def safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")

    if not values.notna().any():
        return None

    return float(values.mean())


def safe_std(series: pd.Series) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return None

    return float(values.std(ddof=1))


def safe_min(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")

    if not values.notna().any():
        return None

    return float(values.min())


def safe_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")

    if not values.notna().any():
        return None

    return float(values.max())


def _flag_sum(
    df: pd.DataFrame,
    column: str,
) -> int:
    if column not in df.columns:
        return 0

    return int(
        df[column]
        .fillna(False)
        .astype(bool)
        .sum()
    )


def _optional_mean(
    df: pd.DataFrame,
    column: str,
) -> float | None:
    if column not in df.columns:
        return None

    return safe_mean(df[column])


def _contact_outcomes(
    batted: pd.DataFrame,
) -> dict[str, Any]:
    batted_balls = int(len(batted))

    output = {
        "batted_balls": batted_balls,
        "avg_exit_velocity": _optional_mean(
            batted,
            "launch_speed",
        ),
        "max_exit_velocity": (
            safe_max(batted["launch_speed"])
            if "launch_speed" in batted.columns
            else None
        ),
        "avg_launch_angle": _optional_mean(
            batted,
            "launch_angle",
        ),
        "hard_hit_balls": _flag_sum(
            batted,
            "_is_hard_hit",
        ),
        "hard_hit_pct": safe_rate(
            _flag_sum(batted, "_is_hard_hit"),
            batted_balls,
        ),
        "ground_balls": _flag_sum(
            batted,
            "_is_ground_ball",
        ),
        "fly_balls": _flag_sum(
            batted,
            "_is_fly_ball",
        ),
        "line_drives": _flag_sum(
            batted,
            "_is_line_drive",
        ),
        "popups": _flag_sum(
            batted,
            "_is_popup",
        ),
    }

    if "estimated_woba_using_speedangle" in batted.columns:
        output["avg_expected_woba"] = _optional_mean(
            batted,
            "estimated_woba_using_speedangle",
        )

    if "estimated_ba_using_speedangle" in batted.columns:
        output["avg_expected_ba"] = _optional_mean(
            batted,
            "estimated_ba_using_speedangle",
        )

    if "estimated_slg_using_speedangle" in batted.columns:
        output["avg_expected_slg"] = _optional_mean(
            batted,
            "estimated_slg_using_speedangle",
        )

    return output


def summarize_pitcher_evidence(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build factual historical evidence only.

    No predictive grade, importance, confidence, or transferability
    is assumed here.
    """
    if df.empty:
        return {
            "games": 0,
            "appearances": 0,
            "pitches": 0,
            "evidence_status": {
                "candidate_only": True,
                "predictive_importance_assumed": False,
            },
        }

    pitches = int(len(df))
    games = int(df["game_pk"].nunique())

    appearances = int(
        df[
            ["game_pk", "pitcher_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    pa_end = df[
        df["_is_pa_end"].fillna(False)
    ].copy()

    batted = df[
        df["_is_batted_ball"].fillna(False)
    ].copy()

    first_pitch = df[
        df["_is_first_pitch"].fillna(False)
    ].copy()

    two_strike = df[
        df["_is_two_strike_pitch"].fillna(False)
    ].copy()

    three_ball = df[
        df["_is_three_ball_pitch"].fillna(False)
    ].copy()

    full_count = df[
        df["_is_full_count"].fillna(False)
    ].copy()

    swings = _flag_sum(df, "_is_swing")
    whiffs = _flag_sum(df, "_is_whiff")
    called_strikes = _flag_sum(df, "_is_called_strike")
    foul_strikes = _flag_sum(df, "_is_foul")
    strikes_thrown = _flag_sum(df, "_is_strike")
    balls_thrown = _flag_sum(df, "_is_ball")
    zone_pitches = _flag_sum(df, "_is_zone")
    out_zone_pitches = _flag_sum(df, "_is_out_zone")
    chase_swings = _flag_sum(df, "_is_chase")

    summary = {
        "games": games,
        "appearances": appearances,
        "pitches": pitches,

        "outcomes": {
            "plate_appearances_ended": int(len(pa_end)),
            "strikeouts": _flag_sum(
                pa_end,
                "_is_strikeout",
            ),
            "walks": _flag_sum(
                pa_end,
                "_is_walk",
            ),
            "hits_allowed": _flag_sum(
                pa_end,
                "_is_hit",
            ),
            "home_runs_allowed": _flag_sum(
                pa_end,
                "_is_home_run",
            ),
            "strikeout_rate_per_pa": safe_rate(
                _flag_sum(pa_end, "_is_strikeout"),
                len(pa_end),
            ),
            "walk_rate_per_pa": safe_rate(
                _flag_sum(pa_end, "_is_walk"),
                len(pa_end),
            ),
            "hit_rate_per_pa": safe_rate(
                _flag_sum(pa_end, "_is_hit"),
                len(pa_end),
            ),
            "home_run_rate_per_pa": safe_rate(
                _flag_sum(pa_end, "_is_home_run"),
                len(pa_end),
            ),
        },

        "strike_ball_profile": {
            "strikes_thrown": strikes_thrown,
            "balls_thrown": balls_thrown,
            "strike_pct": safe_rate(
                strikes_thrown,
                pitches,
            ),
            "ball_pct": safe_rate(
                balls_thrown,
                pitches,
            ),
            "called_strikes": called_strikes,
            "swinging_strikes": whiffs,
            "foul_strikes": foul_strikes,
            "balls_in_play": _flag_sum(
                df,
                "_is_in_play",
            ),
            "called_strike_pct": safe_rate(
                called_strikes,
                pitches,
            ),
            "swinging_strike_pct_per_pitch": safe_rate(
                whiffs,
                pitches,
            ),
            "whiff_pct_per_swing": safe_rate(
                whiffs,
                swings,
            ),
            "csw_count": called_strikes + whiffs,
            "csw_pct": safe_rate(
                called_strikes + whiffs,
                pitches,
            ),
        },

        "location_profile": {
            "zone_pitches": zone_pitches,
            "zone_pct": safe_rate(
                zone_pitches,
                pitches,
            ),
            "out_zone_pitches": out_zone_pitches,
            "out_zone_pct": safe_rate(
                out_zone_pitches,
                pitches,
            ),
            "chase_swings": chase_swings,
            "chase_pct": safe_rate(
                chase_swings,
                out_zone_pitches,
            ),
            "heart_pitches": _flag_sum(
                df,
                "_is_heart",
            ),
            "heart_pct": safe_rate(
                _flag_sum(df, "_is_heart"),
                pitches,
            ),
            "middle_middle_pitches": _flag_sum(
                df,
                "_is_middle_middle",
            ),
            "middle_middle_pct": safe_rate(
                _flag_sum(df, "_is_middle_middle"),
                pitches,
            ),
            "shadow_pitches": _flag_sum(
                df,
                "_is_shadow",
            ),
            "shadow_pct": safe_rate(
                _flag_sum(df, "_is_shadow"),
                pitches,
            ),
            "edge_pitches": _flag_sum(
                df,
                "_is_edge",
            ),
            "edge_pct": safe_rate(
                _flag_sum(df, "_is_edge"),
                pitches,
            ),
            "waste_pitches": _flag_sum(
                df,
                "_is_waste",
            ),
            "waste_pct": safe_rate(
                _flag_sum(df, "_is_waste"),
                pitches,
            ),
        },

        "count_execution": {
            "first_pitches": int(len(first_pitch)),
            "first_pitch_strikes": _flag_sum(
                first_pitch,
                "_is_strike",
            ),
            "first_pitch_balls": _flag_sum(
                first_pitch,
                "_is_ball",
            ),
            "first_pitch_strike_pct": safe_rate(
                _flag_sum(first_pitch, "_is_strike"),
                len(first_pitch),
            ),

            "two_strike_pitches": int(len(two_strike)),
            "two_strike_strikes": _flag_sum(
                two_strike,
                "_is_strike",
            ),
            "two_strike_balls": _flag_sum(
                two_strike,
                "_is_ball",
            ),
            "two_strike_heart_pitches": _flag_sum(
                two_strike,
                "_is_heart",
            ),
            "two_strike_middle_middle_pitches": _flag_sum(
                two_strike,
                "_is_middle_middle",
            ),
            "two_strike_strikeouts": _flag_sum(
                two_strike,
                "_is_strikeout",
            ),
            "two_strike_whiffs": _flag_sum(
                two_strike,
                "_is_whiff",
            ),
            "two_strike_heart_pct": safe_rate(
                _flag_sum(two_strike, "_is_heart"),
                len(two_strike),
            ),

            "three_ball_pitches": int(len(three_ball)),
            "three_ball_strikes": _flag_sum(
                three_ball,
                "_is_strike",
            ),
            "three_ball_balls": _flag_sum(
                three_ball,
                "_is_ball",
            ),
            "three_ball_heart_pitches": _flag_sum(
                three_ball,
                "_is_heart",
            ),
            "three_ball_middle_middle_pitches": _flag_sum(
                three_ball,
                "_is_middle_middle",
            ),
            "three_ball_heart_pct": safe_rate(
                _flag_sum(three_ball, "_is_heart"),
                len(three_ball),
            ),

            "full_count_pitches": int(len(full_count)),
            "full_count_strikes": _flag_sum(
                full_count,
                "_is_strike",
            ),
            "full_count_balls": _flag_sum(
                full_count,
                "_is_ball",
            ),
            "full_count_walks": _flag_sum(
                full_count,
                "_is_walk",
            ),
            "full_count_strikeouts": _flag_sum(
                full_count,
                "_is_strikeout",
            ),
        },

        "pitch_quality": {
            "avg_velocity": _optional_mean(
                df,
                "release_speed",
            ),
            "velocity_std": (
                safe_std(df["release_speed"])
                if "release_speed" in df.columns
                else None
            ),
            "min_velocity": (
                safe_min(df["release_speed"])
                if "release_speed" in df.columns
                else None
            ),
            "max_velocity": (
                safe_max(df["release_speed"])
                if "release_speed" in df.columns
                else None
            ),
            "avg_effective_speed": _optional_mean(
                df,
                "effective_speed",
            ),
            "avg_spin_rate": _optional_mean(
                df,
                "release_spin_rate",
            ),
            "avg_extension": _optional_mean(
                df,
                "release_extension",
            ),
            "avg_horizontal_movement": _optional_mean(
                df,
                "pfx_x",
            ),
            "avg_vertical_movement": _optional_mean(
                df,
                "pfx_z",
            ),
            "avg_release_x": _optional_mean(
                df,
                "release_pos_x",
            ),
            "avg_release_y": _optional_mean(
                df,
                "release_pos_y",
            ),
            "avg_release_z": _optional_mean(
                df,
                "release_pos_z",
            ),
            "avg_plate_x": _optional_mean(
                df,
                "plate_x",
            ),
            "avg_plate_z": _optional_mean(
                df,
                "plate_z",
            ),
            "avg_velocity_change_from_previous": _optional_mean(
                df,
                "velocity_change_from_previous",
            ),
        },

        "contact_allowed": _contact_outcomes(
            batted
        ),

        "situation_scope": {
            "roles": sorted(
                str(value)
                for value in df["role"].dropna().unique()
            ),
            "pitch_types": sorted(
                str(value)
                for value in df["pitch_type"].dropna().unique()
            ),
            "batter_sides": sorted(
                str(value)
                for value in df["batter_side"].dropna().unique()
            ),
            "pitch_count_min": int(
                df["appearance_pitch_count"].min()
            ),
            "pitch_count_max": int(
                df["appearance_pitch_count"].max()
            ),
        },

        "evidence_status": {
            "candidate_only": True,
            "predictive_importance_assumed": False,
            "single_metric_control_allowed": False,
            "cross_pitcher_transfer_assumed": False,
            "cross_team_transfer_assumed": False,
            "requires_local_validation": True,
        },
    }

    return summary


def grouped_summaries(
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    if column not in df.columns:
        return {}

    output = {}

    for value, group in df.groupby(
        column,
        dropna=True,
        observed=True,
        sort=True,
    ):
        output[str(value)] = summarize_pitcher_evidence(
            group
        )

    return output


def nested_summaries(
    df: pd.DataFrame,
    outer_column: str,
    inner_column: str,
) -> dict[str, Any]:
    if (
        outer_column not in df.columns
        or inner_column not in df.columns
    ):
        return {}

    output = {}

    for outer_value, outer_df in df.groupby(
        outer_column,
        dropna=True,
        observed=True,
        sort=True,
    ):
        output[str(outer_value)] = grouped_summaries(
            outer_df,
            inner_column,
        )

    return output
