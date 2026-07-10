
from pathlib import Path
from datetime import datetime
import json
import re
import pandas as pd

from atlas.config import MASTER_DIR, DATA_DIR
from atlas.memory.memory_engine import save_json_card
from atlas.registry.registry_engine import update_counts, update_engine_status

PLAYER_ENGINE_VERSION = "1.0.0"
PLAYER_CARD_VERSION = "1.0.0"

PLAYER_DIR = DATA_DIR / "history" / "players"


def load_master_pitches():
    path = MASTER_DIR / "master_pitch_database.parquet"
    return pd.read_parquet(path)


def safe_filename(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return value.strip("_")


def build_player_pitch_table(pitches=None):
    if pitches is None:
        pitches = load_master_pitches()

    cols = [
        "game_pk", "game_date", "game_year", "atlas_season", "atlas_game_type",
        "batter", "stand",
        "pitcher", "p_throws",
        "home_team", "away_team",
        "events", "description",
        "pitch_type", "pitch_name",
        "release_speed",
        "launch_speed", "launch_angle",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "woba_value", "woba_denom",
        "babip_value", "iso_value",
        "balls", "strikes",
        "at_bat_number", "pitch_number",
        "inning", "inning_topbot",
        "zone", "type"
    ]

    df = pitches[cols].copy()

    df = df.rename(columns={
        "batter": "player_id",
        "stand": "bats",
        "pitcher": "pitcher_id",
        "p_throws": "pitcher_throws",
    })

    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])

    return df


def build_player_card(player_df):
    player_df = player_df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    first = player_df.iloc[0]
    pa_df = player_df.dropna(subset=["events"]).copy()
    batted_df = player_df.dropna(subset=["launch_speed"]).copy()

    card = {
        "metadata": {
            "player_card_version": PLAYER_CARD_VERSION,
            "player_id": int(first.player_id),
            "player_name": None,
            "bats": first.bats,
            "throws": None,
            "primary_position": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        },

        "career_summary": {
            "seasons": sorted([int(x) for x in player_df["game_year"].dropna().unique().tolist()]),
            "games_seen": int(player_df["game_pk"].nunique()),
            "plate_appearances": int(len(pa_df)),
            "pitches_seen": int(len(player_df)),
            "hits": int(pa_df["events"].isin(["single", "double", "triple", "home_run"]).sum()),
            "home_runs": int((pa_df["events"] == "home_run").sum()),
            "walks": int(pa_df["events"].isin(["walk", "intent_walk"]).sum()),
            "strikeouts": int((pa_df["events"] == "strikeout").sum()),
        },

        "offense": {},
        "contact_profile": {
            "batted_balls": int(len(batted_df)),
            "avg_exit_velocity": float(batted_df["launch_speed"].mean()) if len(batted_df) else None,
            "max_exit_velocity": float(batted_df["launch_speed"].max()) if len(batted_df) else None,
            "avg_launch_angle": float(batted_df["launch_angle"].mean()) if len(batted_df) else None,
        },

        "discipline_profile": {
            "swings": int(player_df["description"].astype(str).str.contains("swing|foul|hit_into_play", case=False, na=False).sum()),
            "whiffs": int(player_df["description"].astype(str).str.contains("swinging_strike", case=False, na=False).sum()),
            "called_strikes": int((player_df["description"] == "called_strike").sum()),
            "balls": int(player_df["type"].eq("B").sum()),
        },

        "timeline": [],
        "baserunning": {},
        "defense": {},
        "environment": {},
        "matchups": {},
        "relationships": {},
        "identity": {
            "identity_version": "empty",
            "confidence": None,
            "games_observed": int(player_df["game_pk"].nunique()),
            "last_updated": None,
        },
        "identity_history": [],
        "learning": {},

        "provenance": {
            "source": "master_pitch_database",
            "player_engine_version": PLAYER_ENGINE_VERSION,
            "built_at": datetime.now().isoformat(),
        },
    }

    for game_pk, g in player_df.groupby("game_pk"):
        season = int(g["game_year"].iloc[0])
        card["timeline"].append({
            "game_pk": int(game_pk),
            "game_card": f"data/history/game_cards/{season}/gamecard_{int(game_pk)}.json",
            "date": g["game_date"].iloc[0].date().isoformat(),
            "season": season,
            "pitches_seen": int(len(g)),
            "plate_appearances": int(g["events"].notna().sum()),
            "pitchers_faced": sorted([int(x) for x in g["pitcher_id"].dropna().unique().tolist()]),
        })

    return card


def save_player_card(card, output_dir=None):
    if output_dir is None:
        output_dir = PLAYER_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    player_id = card["metadata"]["player_id"]
    filename = f"{player_id}.json"
    path = output_dir / filename

    return save_json_card(card, path)


def run_player_engine(limit=None):
    pitches = load_master_pitches()
    player_table = build_player_pitch_table(pitches)

    saved_paths = []

    grouped = player_table.groupby("player_id")

    for i, (player_id, player_df) in enumerate(grouped):
        if limit is not None and i >= limit:
            break

        card = build_player_card(player_df)
        path = save_player_card(card)
        saved_paths.append(path)

    cards_built = len(saved_paths)

    if limit is None:
        update_counts(player_cards=cards_built)
        update_engine_status("player_engine", "complete_v1")
    else:
        update_engine_status("player_engine", "testing_v1")

    summary = {
        "engine": "ATLAS Player Engine",
        "engine_version": PLAYER_ENGINE_VERSION,
        "master_pitch_rows": int(len(pitches)),
        "unique_players": int(player_table["player_id"].nunique()),
        "cards_built": int(cards_built),
        "output_directory": str(PLAYER_DIR),
    }

    print("=" * 60)
    print("ATLAS PLAYER ENGINE")
    print("=" * 60)
    print(f"Master Pitch Rows... {summary['master_pitch_rows']:,}")
    print(f"Unique Players...... {summary['unique_players']:,}")
    print(f"Player Cards Built.. {summary['cards_built']:,}")
    print(f"Saved To............ {summary['output_directory']}")
    print("=" * 60)

    return summary
