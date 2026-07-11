
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import MASTER_DIR, DATA_DIR
from atlas.registry.registry_engine import (
    update_counts,
    update_engine_status,
)


PLAYER_ENGINE_VERSION = "2.0.0"
PLAYER_CARD_VERSION = "2.0.0"

PLAYER_DIR = DATA_DIR / "history" / "players"
MASTER_PITCH_PATH = MASTER_DIR / "master_pitch_database.parquet"

GAME_CARD_ROOT = DATA_DIR / "history" / "game_cards"
GAME_CARD_MANIFEST = GAME_CARD_ROOT / "game_card_manifest.parquet"


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


def _safe_rate(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator in [0, None] or pd.isna(denominator):
        return None
    return float(numerator) / float(denominator)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(".json.tmp")

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            _json_safe(payload),
            file,
            indent=2,
        )

    temporary.replace(destination)
    return destination


def load_master_pitches() -> pd.DataFrame:
    if not MASTER_PITCH_PATH.exists():
        raise FileNotFoundError(
            f"Master pitch database missing: {MASTER_PITCH_PATH}"
        )

    pitches = pd.read_parquet(MASTER_PITCH_PATH)

    if "game_type" not in pitches.columns:
        raise KeyError("master_pitch_database is missing game_type")

    game_types = set(
        pitches["game_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    if game_types != {"R"}:
        raise ValueError(
            "Player Engine requires regular-season-only data. "
            f"Found game types: {sorted(game_types)}"
        )

    return pitches


def build_player_pitch_table(
    pitches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pitches is None:
        pitches = load_master_pitches()

    required_columns = [
        "game_pk",
        "game_date",
        "atlas_season",
        "game_type",
        "batter",
        "stand",
        "pitcher",
        "p_throws",
        "home_team",
        "away_team",
        "events",
        "description",
        "pitch_type",
        "release_speed",
        "launch_speed",
        "launch_angle",
        "balls",
        "strikes",
        "at_bat_number",
        "pitch_number",
        "inning",
        "inning_topbot",
        "zone",
        "type",
    ]

    missing = [
        column
        for column in required_columns
        if column not in pitches.columns
    ]

    if missing:
        raise KeyError(
            f"Player Engine missing required columns: {missing}"
        )

    optional_columns = [
        "game_year",
        "atlas_game_type",
        "pitch_name",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "launch_speed_angle",
        "hc_x",
        "hc_y",
        "bb_type",
        "effective_speed",
        "release_spin_rate",
        "release_extension",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "bat_score",
        "fld_score",
        "post_bat_score",
        "post_fld_score",
        "delta_home_win_exp",
        "delta_run_exp",
        "estimated_slg_using_speedangle",
        "estimated_woba_using_speedangle",
        "if_fielding_alignment",
        "of_fielding_alignment",
    ]

    selected_columns = required_columns + [
        column
        for column in optional_columns
        if column in pitches.columns
        and column not in required_columns
    ]

    # Preserve order while preventing duplicate column names.
    selected_columns = list(dict.fromkeys(selected_columns))

    table = pitches[selected_columns].copy()

    table = table.rename(
        columns={
            "batter": "player_id",
            "stand": "bats",
            "pitcher": "pitcher_id",
            "p_throws": "pitcher_throws",
        }
    )

    table = table.dropna(subset=["player_id"])
    table["player_id"] = table["player_id"].astype("int64")
    table["game_pk"] = table["game_pk"].astype("int64")
    table["game_date"] = pd.to_datetime(
        table["game_date"],
        errors="coerce",
    )

    table["batting_team"] = np.where(
        table["inning_topbot"].astype(str).eq("Top"),
        table["away_team"],
        table["home_team"],
    )

    table["home_away"] = np.where(
        table["inning_topbot"].astype(str).eq("Top"),
        "AWAY",
        "HOME",
    )

    table["opponent"] = np.where(
        table["inning_topbot"].astype(str).eq("Top"),
        table["home_team"],
        table["away_team"],
    )

    description = table["description"].astype("string")
    events = table["events"].astype("string")

    table["_is_pa"] = table["events"].notna()
    table["_is_hit"] = events.isin(HIT_EVENTS)
    table["_is_single"] = events.eq("single")
    table["_is_double"] = events.eq("double")
    table["_is_triple"] = events.eq("triple")
    table["_is_home_run"] = events.eq("home_run")
    table["_is_walk"] = events.isin(WALK_EVENTS)
    table["_is_strikeout"] = events.isin(STRIKEOUT_EVENTS)
    table["_is_hbp"] = events.eq("hit_by_pitch")
    # Statcast pitch result type "X" identifies a ball put into play.
    # Requiring launch_speed preserves measurable contact quality
    # while excluding tracked foul-ball contact.
    table["_is_batted_ball"] = (
        table["type"].astype("string").eq("X")
        & table["launch_speed"].notna()
    )

    table["_is_swing"] = description.isin(SWING_DESCRIPTIONS)
    table["_is_whiff"] = description.isin(WHIFF_DESCRIPTIONS)
    table["_is_called_strike"] = description.eq("called_strike")
    table["_is_ball"] = table["type"].astype("string").eq("B")

    zone_numeric = pd.to_numeric(
        table["zone"],
        errors="coerce",
    )

    table["_is_in_zone"] = zone_numeric.between(1, 9)
    table["_is_out_zone"] = zone_numeric.between(11, 14)
    table["_is_chase"] = (
        table["_is_swing"]
        & table["_is_out_zone"]
    )

    balls_numeric = pd.to_numeric(
        table["balls"],
        errors="coerce",
    )
    strikes_numeric = pd.to_numeric(
        table["strikes"],
        errors="coerce",
    )

    table["count_state"] = np.select(
        [
            strikes_numeric.eq(2),
            balls_numeric.eq(3),
            balls_numeric.gt(strikes_numeric),
            strikes_numeric.gt(balls_numeric),
        ],
        [
            "two_strike",
            "three_ball",
            "ahead",
            "behind",
        ],
        default="even",
    )

    return table


def _summarize_split(
    split_df: pd.DataFrame,
) -> dict[str, Any]:
    if split_df.empty:
        return {
            "games": 0,
            "plate_appearances": 0,
            "pitches_seen": 0,
        }

    pa_df = split_df[split_df["_is_pa"]]
    batted_df = split_df[split_df["_is_batted_ball"]]

    pitches_seen = int(len(split_df))
    plate_appearances = int(len(pa_df))

    swings = int(split_df["_is_swing"].sum())
    whiffs = int(split_df["_is_whiff"].sum())
    called_strikes = int(
        split_df["_is_called_strike"].sum()
    )
    balls_seen = int(split_df["_is_ball"].sum())
    chase_swings = int(split_df["_is_chase"].sum())
    out_zone_pitches = int(
        split_df["_is_out_zone"].sum()
    )

    hits = int(pa_df["_is_hit"].sum())
    home_runs = int(pa_df["_is_home_run"].sum())
    walks = int(pa_df["_is_walk"].sum())
    strikeouts = int(pa_df["_is_strikeout"].sum())

    hard_hit_balls = int(
        (
            pd.to_numeric(
                batted_df["launch_speed"],
                errors="coerce",
            ) >= 95.0
        ).sum()
    )

    summary = {
        "games": int(split_df["game_pk"].nunique()),
        "plate_appearances": plate_appearances,
        "pitches_seen": pitches_seen,

        "outcomes": {
            "hits": hits,
            "singles": int(pa_df["_is_single"].sum()),
            "doubles": int(pa_df["_is_double"].sum()),
            "triples": int(pa_df["_is_triple"].sum()),
            "home_runs": home_runs,
            "walks": walks,
            "strikeouts": strikeouts,
            "hit_by_pitch": int(pa_df["_is_hbp"].sum()),
            "hit_rate_per_pa": _safe_rate(
                hits,
                plate_appearances,
            ),
            "home_run_rate_per_pa": _safe_rate(
                home_runs,
                plate_appearances,
            ),
            "walk_rate_per_pa": _safe_rate(
                walks,
                plate_appearances,
            ),
            "strikeout_rate_per_pa": _safe_rate(
                strikeouts,
                plate_appearances,
            ),
        },

        "discipline": {
            "swings": swings,
            "whiffs": whiffs,
            "called_strikes": called_strikes,
            "balls": balls_seen,
            "chase_swings": chase_swings,
            "out_zone_pitches": out_zone_pitches,
            "swing_pct": _safe_rate(
                swings,
                pitches_seen,
            ),
            "whiff_pct_per_swing": _safe_rate(
                whiffs,
                swings,
            ),
            "called_strike_pct": _safe_rate(
                called_strikes,
                pitches_seen,
            ),
            "ball_pct": _safe_rate(
                balls_seen,
                pitches_seen,
            ),
            "chase_pct": _safe_rate(
                chase_swings,
                out_zone_pitches,
            ),
        },

        "contact": {
            "batted_balls": int(len(batted_df)),
            "avg_exit_velocity": (
                float(
                    pd.to_numeric(
                        batted_df["launch_speed"],
                        errors="coerce",
                    ).mean()
                )
                if not batted_df.empty
                else None
            ),
            "max_exit_velocity": (
                float(
                    pd.to_numeric(
                        batted_df["launch_speed"],
                        errors="coerce",
                    ).max()
                )
                if not batted_df.empty
                else None
            ),
            "avg_launch_angle": (
                float(
                    pd.to_numeric(
                        batted_df["launch_angle"],
                        errors="coerce",
                    ).mean()
                )
                if not batted_df.empty
                else None
            ),
            "hard_hit_balls": hard_hit_balls,
            "hard_hit_pct": _safe_rate(
                hard_hit_balls,
                len(batted_df),
            ),
        },
    }

    if "woba_value" in pa_df.columns:
        woba_values = pd.to_numeric(
            pa_df["woba_value"],
            errors="coerce",
        )
        summary["outcomes"]["avg_woba_value"] = (
            float(woba_values.mean())
            if woba_values.notna().any()
            else None
        )

    if "estimated_woba_using_speedangle" in batted_df.columns:
        expected_woba = pd.to_numeric(
            batted_df[
                "estimated_woba_using_speedangle"
            ],
            errors="coerce",
        )
        summary["contact"]["avg_expected_woba"] = (
            float(expected_woba.mean())
            if expected_woba.notna().any()
            else None
        )

    return summary


def _build_grouped_summaries(
    player_df: pd.DataFrame,
    group_column: str,
) -> dict[str, dict[str, Any]]:
    if group_column not in player_df.columns:
        return {}

    output = {}

    for value, split_df in player_df.groupby(
        group_column,
        dropna=True,
    ):
        output[str(value)] = _summarize_split(
            split_df
        )

    return output


def _build_game_timeline(
    player_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    timeline = []

    sort_columns = [
        column
        for column in [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ]
        if column in player_df.columns
    ]

    player_df = player_df.sort_values(
        sort_columns,
        kind="stable",
    )

    for game_pk, game_df in player_df.groupby(
        "game_pk",
        sort=False,
    ):
        first = game_df.iloc[0]
        summary = _summarize_split(game_df)

        timeline.append({
            "game_pk": int(game_pk),
            "date": (
                first["game_date"].date().isoformat()
                if pd.notna(first["game_date"])
                else None
            ),
            "season": int(first["atlas_season"]),
            "team": first.get("batting_team"),
            "opponent": first.get("opponent"),
            "home_away": first.get("home_away"),
            "bats": first.get("bats"),
            "pitchers_faced": sorted(
                int(value)
                for value in game_df[
                    "pitcher_id"
                ].dropna().unique()
            ),
            "summary": summary,
            "game_card_reference": {
                "game_pk": int(game_pk),
                "manifest": str(
                    GAME_CARD_MANIFEST
                ),
                "event_store_season": int(
                    first["atlas_season"]
                ),
                "lookup_key": str(int(game_pk)),
            },
        })

    return timeline


def build_player_card(
    player_df: pd.DataFrame,
) -> dict[str, Any]:
    if player_df.empty:
        raise ValueError("player_df is empty")

    player_df = player_df.copy()

    first = player_df.sort_values(
        ["game_date", "game_pk"],
        kind="stable",
    ).iloc[0]

    player_id = int(first["player_id"])

    seasons = sorted(
        int(value)
        for value in player_df[
            "atlas_season"
        ].dropna().unique()
    )

    teams = sorted(
        str(value)
        for value in player_df[
            "batting_team"
        ].dropna().unique()
    )

    overall = _summarize_split(player_df)

    card = {
        "metadata": {
            "player_card_version": PLAYER_CARD_VERSION,
            "player_engine_version": PLAYER_ENGINE_VERSION,
            "player_id": player_id,
            "player_name": None,
            "bats": first.get("bats"),
            "created_at": (
                datetime.now(timezone.utc).isoformat()
            ),
            "updated_at": (
                datetime.now(timezone.utc).isoformat()
            ),
            "regular_season_only": True,
            "game_type_filter": "R",
        },

        "scope": {
            "seasons": seasons,
            "teams": teams,
            "date_start": (
                player_df["game_date"]
                .min()
                .date()
                .isoformat()
            ),
            "date_end": (
                player_df["game_date"]
                .max()
                .date()
                .isoformat()
            ),
        },

        "overall": overall,

        "season_identities": _build_grouped_summaries(
            player_df,
            "atlas_season",
        ),

        "team_identities": _build_grouped_summaries(
            player_df,
            "batting_team",
        ),

        "pitcher_handedness_identities":
            _build_grouped_summaries(
                player_df,
                "pitcher_throws",
            ),

        "pitch_type_identities":
            _build_grouped_summaries(
                player_df,
                "pitch_type",
            ),

        "count_state_identities":
            _build_grouped_summaries(
                player_df,
                "count_state",
            ),

        "home_away_identities":
            _build_grouped_summaries(
                player_df,
                "home_away",
            ),

        "candidate_contexts": {
            "measured_not_assumed_predictive": True,
            "prediction_weights_assigned": False,
            "requires_evidence_validation": True,
        },

        "timeline": _build_game_timeline(
            player_df
        ),

        "identity": {
            "current_identity_version": None,
            "identity_transition_confirmed": False,
            "current_team": (
                player_df.sort_values(
                    "game_date",
                    kind="stable",
                )["batting_team"]
                .dropna()
                .iloc[-1]
                if player_df[
                    "batting_team"
                ].notna().any()
                else None
            ),
            "current_season": max(seasons),
            "games_observed": overall["games"],
            "identity_history": [],
        },

        "evidence_links": [],
        "validation_links": [],
        "transition_links": [],

        "traceability": {
            "source_dataset": str(
                MASTER_PITCH_PATH
            ),
            "game_card_manifest": str(
                GAME_CARD_MANIFEST
            ),
            "source_game_count": int(
                player_df["game_pk"].nunique()
            ),
            "source_event_rows": int(
                len(player_df)
            ),
            "source_game_pks": sorted(
                int(value)
                for value in player_df[
                    "game_pk"
                ].dropna().unique()
            ),
            "full_events_stored_in_game_cards": True,
            "event_rows_embedded_in_card": False,
        },

        "data_quality": {
            "missing_player_id_rows": int(
                player_df[
                    "player_id"
                ].isna().sum()
            ),
            "duplicate_pitch_rows_not_removed": True,
            "regular_season_verified": True,
        },

        "provenance": {
            "source": (
                "clean regular-season "
                "master_pitch_database.parquet"
            ),
            "built_at": (
                datetime.now(timezone.utc).isoformat()
            ),
        },
    }

    return card


def save_player_card(
    card: dict[str, Any],
    output_dir: Path | str | None = None,
) -> Path:
    if output_dir is None:
        output_dir = PLAYER_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_id = int(
        card["metadata"]["player_id"]
    )

    return _atomic_json_write(
        card,
        output_dir / f"{player_id}.json",
    )


def run_player_engine(
    limit: int | None = None,
) -> dict[str, Any]:
    pitches = load_master_pitches()
    player_table = build_player_pitch_table(
        pitches
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    staging_dir = (
        PLAYER_DIR.parent
        / f"players_build_{timestamp}"
    )

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    grouped = player_table.groupby(
        "player_id",
        sort=True,
    )

    cards_built = 0
    total_events_written = 0
    total_games_linked = 0

    try:
        for index, (
            player_id,
            player_df,
        ) in enumerate(grouped):
            if limit is not None and index >= limit:
                break

            card = build_player_card(
                player_df
            )

            save_player_card(
                card,
                output_dir=staging_dir,
            )

            cards_built += 1
            total_events_written += int(
                card["traceability"][
                    "source_event_rows"
                ]
            )
            total_games_linked += int(
                card["traceability"][
                    "source_game_count"
                ]
            )

        if cards_built == 0:
            raise RuntimeError(
                "Player Engine built zero cards."
            )

        expected_cards = (
            min(
                int(
                    player_table[
                        "player_id"
                    ].nunique()
                ),
                limit,
            )
            if limit is not None
            else int(
                player_table[
                    "player_id"
                ].nunique()
            )
        )

        if cards_built != expected_cards:
            raise AssertionError(
                f"Expected {expected_cards} cards; "
                f"built {cards_built}."
            )

        if limit is None:
            backup_dir = (
                PLAYER_DIR.parent
                / "backups"
                / f"players_pre_v2_{timestamp}"
            )

            if PLAYER_DIR.exists():
                backup_dir.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                PLAYER_DIR.replace(
                    backup_dir
                )

            staging_dir.replace(
                PLAYER_DIR
            )

            update_counts(
                player_cards=cards_built
            )
            update_engine_status(
                "player_engine",
                "complete_v2",
            )

            final_output_dir = PLAYER_DIR
        else:
            update_engine_status(
                "player_engine",
                "testing_v2",
            )
            final_output_dir = staging_dir

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    summary = {
        "engine": "ATLAS Player Engine",
        "engine_version": PLAYER_ENGINE_VERSION,
        "card_version": PLAYER_CARD_VERSION,
        "master_pitch_rows": int(
            len(pitches)
        ),
        "player_event_rows": int(
            len(player_table)
        ),
        "unique_players": int(
            player_table[
                "player_id"
            ].nunique()
        ),
        "cards_built": int(
            cards_built
        ),
        "total_card_event_references": int(
            total_events_written
        ),
        "total_game_references": int(
            total_games_linked
        ),
        "regular_season_only": True,
        "output_directory": str(
            final_output_dir
        ),
    }

    print("=" * 72)
    print("ATLAS PLAYER ENGINE V2")
    print("=" * 72)
    print(
        f"Master Pitch Rows....... "
        f"{summary['master_pitch_rows']:,}"
    )
    print(
        f"Player Event Rows....... "
        f"{summary['player_event_rows']:,}"
    )
    print(
        f"Unique Players.......... "
        f"{summary['unique_players']:,}"
    )
    print(
        f"Player Cards Built...... "
        f"{summary['cards_built']:,}"
    )
    print(
        f"Event References........ "
        f"{summary['total_card_event_references']:,}"
    )
    print(
        f"Saved To................ "
        f"{summary['output_directory']}"
    )
    print("=" * 72)

    return summary
