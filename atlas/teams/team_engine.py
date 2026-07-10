
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from atlas.teams.team_schema import blank_team_card


# ============================================================
# BUILD TEAM CARD
# ============================================================

def build_team_card(team_df, team):

    card = blank_team_card()

    card["metadata"] = {
        "team": team,
        "team_card_version": "1.0.0",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

    card["career_summary"] = {
        "games": int(len(team_df)),
        "seasons": sorted(team_df["atlas_season"].dropna().unique().tolist())
        if "atlas_season" in team_df.columns else [],
    }

    timeline = []

    for _, row in team_df.sort_values("game_date").iterrows():

        timeline.append({

            "game_pk": row.get("game_pk"),
            "date": str(row.get("game_date")),
            "opponent": row.get("opponent"),
            "home_away": row.get("home_away"),

            "runs_scored": row.get("runs_scored"),
            "runs_allowed": row.get("runs_allowed"),

            "won": bool(row.get("won")),

        })

    card["timeline"] = timeline

    card["provenance"] = {
        "team_engine_version": "1.0.0",
        "built_at": datetime.utcnow().isoformat(),
        "source": "team_game_state"
    }

    return card


# ============================================================
# BUILD ALL TEAMS
# ============================================================

def build_all_team_cards(team_game_state, output_dir, limit=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teams = sorted(team_game_state.team.unique())

    if limit:
        teams = teams[:limit]

    built = 0

    for team in teams:

        team_df = team_game_state[
            team_game_state.team == team
        ].copy()

        card = build_team_card(team_df, team)

        with open(output_dir / f"{team}.json", "w") as f:
            json.dump(card, f, indent=2)

        built += 1

    print("=" * 60)
    print("ATLAS TEAM ENGINE")
    print("=" * 60)
    print(f"Teams Built : {built}")
    print(f"Saved To    : {output_dir}")

    return {
        "teams": built,
        "output_directory": str(output_dir)
    }
