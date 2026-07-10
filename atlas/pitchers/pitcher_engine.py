from pathlib import Path
from datetime import datetime
import json
import pandas as pd

from atlas.config import MASTER_DIR, DATA_DIR

PITCHER_ENGINE_VERSION = "1.0.0"
PITCHER_CARD_VERSION = "1.1.0"

PITCHER_DIR = DATA_DIR / "history" / "pitchers"


def load_master_games():
    path = MASTER_DIR / "master_game_database.parquet"
    games = pd.read_parquet(path)
    return games


def build_pitcher_table(games=None):
    if games is None:
        games = load_master_games()

    home = games[[
        "game_pk", "game_date", "game_year",
        "home_team", "away_team",
        "home_starter_id", "home_starter_name", "home_starter_throws",
        "home_starter_pitches", "home_starter_strikeouts",
        "home_starter_walks", "home_starter_hits_allowed",
        "home_starter_runs_allowed"
    ]].copy()

    home.columns = [
        "game_pk", "game_date", "season",
        "team", "opponent",
        "pitcher_id", "pitcher_name", "throws",
        "pitches", "strikeouts", "walks",
        "hits_allowed", "runs_allowed"
    ]

    away = games[[
        "game_pk", "game_date", "game_year",
        "away_team", "home_team",
        "away_starter_id", "away_starter_name", "away_starter_throws",
        "away_starter_pitches", "away_starter_strikeouts",
        "away_starter_walks", "away_starter_hits_allowed",
        "away_starter_runs_allowed"
    ]].copy()

    away.columns = home.columns

    starters = pd.concat([home, away], ignore_index=True)
    starters = starters.dropna(subset=["pitcher_id"])
    starters["pitcher_id"] = starters["pitcher_id"].astype(int)
    starters["game_date"] = pd.to_datetime(starters["game_date"])

    starters = starters.sort_values(["pitcher_id", "game_date"])

    return starters


def build_pitcher_card(pitcher_df):
    pitcher_df = pitcher_df.sort_values("game_date")
    first = pitcher_df.iloc[0]

    card = {
        "metadata": {
            "pitcher_card_version": PITCHER_CARD_VERSION,
            "pitcher_id": int(first.pitcher_id),
            "pitcher_name": first.pitcher_name,
            "throws": first.throws,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        },

        "career_summary": {
            "games_started": int(len(pitcher_df)),
            "teams": sorted(pitcher_df.team.dropna().unique().tolist()),
            "seasons": sorted([int(x) for x in pitcher_df.season.dropna().unique().tolist()]),

            "career_totals": {
                "pitches": int(pitcher_df.pitches.fillna(0).sum()),
                "strikeouts": int(pitcher_df.strikeouts.fillna(0).sum()),
                "walks": int(pitcher_df.walks.fillna(0).sum()),
                "hits_allowed": int(pitcher_df.hits_allowed.fillna(0).sum()),
                "runs_allowed": int(pitcher_df.runs_allowed.fillna(0).sum()),
            },
        },

        "timeline": [],

        "velocity_profile": {},
        "pitch_mix": {},
        "park_splits": {},
        "home_road_splits": {},
        "weather_splits": {},
        "rest_splits": {},
        "lineup_matchups": {},

        # Pitcher Card v1.1 placeholders
        "inning_profile": {},
        "first_inning_profile": {},
        "times_through_order": {},
        "pitch_count_buckets": {},
        "velocity_by_inning": {},
        "whiffs_by_inning": {},
        "yrfi_nrfi_profile": {},

        "identity": {
            "identity_version": "empty",
            "confidence": None,
            "games_observed": int(len(pitcher_df)),
            "last_updated": None,
        },

        "learning": {},

        "provenance": {
            "source": "master_game_database",
            "pitcher_engine_version": PITCHER_ENGINE_VERSION,
            "built_at": datetime.now().isoformat(),
        },
    }

    for _, row in pitcher_df.iterrows():
        card["timeline"].append({
            "game_pk": int(row.game_pk),
            "date": row.game_date.date().isoformat(),
            "season": int(row.season),
            "team": row.team,
            "opponent": row.opponent,
            "pitches": int(row.pitches) if pd.notna(row.pitches) else None,
            "strikeouts": int(row.strikeouts) if pd.notna(row.strikeouts) else None,
            "walks": int(row.walks) if pd.notna(row.walks) else None,
            "hits_allowed": int(row.hits_allowed) if pd.notna(row.hits_allowed) else None,
            "runs_allowed": int(row.runs_allowed) if pd.notna(row.runs_allowed) else None,
        })

    return card


def save_pitcher_card(card, output_dir=None):
    if output_dir is None:
        output_dir = PITCHER_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pitcher_id = card["metadata"]["pitcher_id"]
    path = output_dir / f"{pitcher_id}.json"

    with open(path, "w") as f:
        json.dump(card, f, indent=2)

    return path


def run_pitcher_engine(limit=None):
    games = load_master_games()
    starters = build_pitcher_table(games)

    saved_paths = []

    grouped = starters.groupby("pitcher_id")

    for i, (pitcher_id, pitcher_df) in enumerate(grouped):
        if limit is not None and i >= limit:
            break

        card = build_pitcher_card(pitcher_df)
        path = save_pitcher_card(card)
        saved_paths.append(path)

    summary = {
        "engine": "ATLAS Pitcher Engine",
        "engine_version": PITCHER_ENGINE_VERSION,
        "master_game_rows": int(len(games)),
        "starter_appearances": int(len(starters)),
        "unique_pitchers": int(starters["pitcher_id"].nunique()),
        "cards_built": int(len(saved_paths)),
        "output_directory": str(PITCHER_DIR),
    }

    print("=" * 60)
    print("ATLAS PITCHER ENGINE")
    print("=" * 60)
    print(f"Master Game Rows..... {summary['master_game_rows']:,}")
    print(f"Starter Appearances.. {summary['starter_appearances']:,}")
    print(f"Unique Pitchers...... {summary['unique_pitchers']:,}")
    print(f"Pitcher Cards Built.. {summary['cards_built']:,}")
    print(f"Saved To............. {summary['output_directory']}")
    print("=" * 60)

    return summary
