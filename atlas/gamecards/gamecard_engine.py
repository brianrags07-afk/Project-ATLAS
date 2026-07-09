
from datetime import datetime
from zoneinfo import ZoneInfo

from atlas.config import ATLAS_VERSION, GAMECARD_DIR, today_str
from atlas.utils.files import save_json


def now_iso():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat()


def build_base_game_card(game, starters=None, lineups=None):
    """
    Build one ATLAS Game Card from MLB schedule data plus optional starters/lineups.
    """

    game_pk = game["gamePk"]

    away_team = game["teams"]["away"]["team"]
    home_team = game["teams"]["home"]["team"]

    away_pitcher = game["teams"]["away"].get("probablePitcher")
    home_pitcher = game["teams"]["home"].get("probablePitcher")

    card = {
        "game_pk": game_pk,
        "game_date": game.get("gameDate"),
        "venue": game.get("venue", {}).get("name"),
        "mlb_status": game.get("status", {}).get("detailedState"),

        "teams": {
            "away": {
                "team_id": away_team.get("id"),
                "name": away_team.get("name"),
                "abbr": away_team.get("abbreviation"),
                "probable_pitcher": {
                    "id": away_pitcher.get("id") if away_pitcher else None,
                    "name": away_pitcher.get("fullName") if away_pitcher else "TBD",
                    "ready": away_pitcher is not None,
                },
                "lineup": [],
            },
            "home": {
                "team_id": home_team.get("id"),
                "name": home_team.get("name"),
                "abbr": home_team.get("abbreviation"),
                "probable_pitcher": {
                    "id": home_pitcher.get("id") if home_pitcher else None,
                    "name": home_pitcher.get("fullName") if home_pitcher else "TBD",
                    "ready": home_pitcher is not None,
                },
                "lineup": [],
            },
        },

        "readiness": {
            "away_starter_ready": away_pitcher is not None,
            "home_starter_ready": home_pitcher is not None,
            "away_lineup_ready": False,
            "home_lineup_ready": False,
            "pregame_status": "PARTIAL",
            "warnings": [],
        },

        "atlas": {
            "version": ATLAS_VERSION,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "engine_history": ["gamecard_engine.build_base_game_card"],
        },

        "layers": {
            "daily_data": {},
            "identity": {},
            "matchup": {},
            "features": {},
            "predictions": {
                "moneyline": None,
                "totals": None,
                "props": {},
            },
            "live": {},
            "postgame": {},
            "learning": {},
        },
    }

    if not away_pitcher:
        card["readiness"]["warnings"].append("Missing away probable pitcher")

    if not home_pitcher:
        card["readiness"]["warnings"].append("Missing home probable pitcher")

    return card


def attach_lineups(card, lineups):
    """
    Attach confirmed lineup rows from lineups DataFrame to a Game Card.
    """

    game_pk = card["game_pk"]

    for side in ["away", "home"]:
        team_name = card["teams"][side]["name"]
        team_rows = lineups[
            (lineups["game_pk"] == game_pk) &
            (lineups["team"] == team_name)
        ].copy()

        if len(team_rows) > 0:
            team_rows = team_rows.sort_values("batting_order")

        lineup = []

        for _, row in team_rows.iterrows():
            lineup.append({
                "batting_order": int(row["batting_order"]),
                "player_id": int(row["player_id"]),
                "player_name": row["player_name"],
                "bat_side": row["bat_side"],
                "throw_side": row["throw_side"],
                "position": row["position"],
            })

        card["teams"][side]["lineup"] = lineup
        card["readiness"][f"{side}_lineup_ready"] = len(lineup) >= 9

        if len(lineup) < 9:
            card["readiness"]["warnings"].append(
                f"Missing or incomplete {side} lineup"
            )

    required_ready = (
        card["readiness"]["away_starter_ready"] and
        card["readiness"]["home_starter_ready"] and
        card["readiness"]["away_lineup_ready"] and
        card["readiness"]["home_lineup_ready"]
    )

    card["readiness"]["pregame_status"] = "READY" if required_ready else "PARTIAL"
    card["atlas"]["updated_at"] = now_iso()
    card["atlas"]["engine_history"].append("gamecard_engine.attach_lineups")

    return card


def save_game_card(card, date=None):
    """
    Save a Game Card JSON file.
    """

    if date is None:
        date = today_str()

    out_dir = GAMECARD_DIR / date
    out_path = out_dir / f"gamecard_{card['game_pk']}.json"

    save_json(card, out_path)

    return out_path


def build_and_save_game_cards(games, lineups=None, date=None):
    """
    Build and save one Game Card per game.
    """

    cards = []
    paths = []

    for game in games:
        card = build_base_game_card(game)

        if lineups is not None:
            card = attach_lineups(card, lineups)

        path = save_game_card(card, date=date)

        cards.append(card)
        paths.append(path)

    return cards, paths
