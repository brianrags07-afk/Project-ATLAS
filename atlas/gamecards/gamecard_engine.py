from atlas.config import GAMECARD_DIR, today_str
from atlas.utils.files import save_json

GAME_CARD_VERSION = "2.0.0"


def _lineup_for_game(lineups, game_pk, team_name):
    rows = lineups[
        (lineups["game_pk"] == game_pk) &
        (lineups["team"] == team_name)
    ].copy()

    if len(rows):
        rows = rows.sort_values("batting_order")

    return rows.to_dict("records")


def create_game_card(game, daily):
    game_pk = game["gamePk"]
    date = daily["date"]

    away = game["teams"]["away"]
    home = game["teams"]["home"]

    away_team = away["team"]
    home_team = home["team"]

    away_pitcher = away.get("probablePitcher")
    home_pitcher = home.get("probablePitcher")

    away_lineup = _lineup_for_game(
        daily["lineups"], game_pk, away_team["name"]
    )
    home_lineup = _lineup_for_game(
        daily["lineups"], game_pk, home_team["name"]
    )

    card_id = f"{date}_{game_pk}"

    away_lineup_ready = len(away_lineup) >= 9
    home_lineup_ready = len(home_lineup) >= 9
    away_starter_ready = away_pitcher is not None
    home_starter_ready = home_pitcher is not None

    ready = (
        away_lineup_ready and home_lineup_ready
        and away_starter_ready and home_starter_ready
    )

    warnings = []
    if not away_starter_ready:
        warnings.append("Missing away starter")
    if not home_starter_ready:
        warnings.append("Missing home starter")
    if not away_lineup_ready:
        warnings.append("Missing or incomplete away lineup")
    if not home_lineup_ready:
        warnings.append("Missing or incomplete home lineup")

    return {
        "metadata": {
            "game_card_version": GAME_CARD_VERSION,
            "card_id": card_id,
            "game_pk": game_pk,
            "date": date,
            "mlb_status": game.get("status", {}).get("detailedState"),
        },

        "pregame": {
            "venue": game.get("venue", {}).get("name"),
            "away_team": {
                "team_id": away_team.get("id"),
                "name": away_team.get("name"),
                "abbr": away_team.get("abbreviation"),
                "starter": {
                    "id": away_pitcher.get("id") if away_pitcher else None,
                    "name": away_pitcher.get("fullName") if away_pitcher else "TBD",
                },
                "lineup": away_lineup,
            },
            "home_team": {
                "team_id": home_team.get("id"),
                "name": home_team.get("name"),
                "abbr": home_team.get("abbreviation"),
                "starter": {
                    "id": home_pitcher.get("id") if home_pitcher else None,
                    "name": home_pitcher.get("fullName") if home_pitcher else "TBD",
                },
                "lineup": home_lineup,
            },
        },

        "readiness": {
            "away_starter_ready": away_starter_ready,
            "home_starter_ready": home_starter_ready,
            "away_lineup_ready": away_lineup_ready,
            "home_lineup_ready": home_lineup_ready,
            "pregame_status": "READY" if ready else "PARTIAL",
            "warnings": warnings,
        },

        "series": {},
        "travel": {},
        "bullpen": {},
        "weather": {},
        "umpire": {},
        "identity": {},
        "matchup": {},
        "features": {},
        "prediction": {},
        "live": {},
        "postgame": {},
        "learning": {},
    }


def build_game_cards(daily):
    cards = []
    for game in daily["games"]:
        cards.append(create_game_card(game, daily))
    return cards


def save_game_cards(cards, date=None):
    if date is None:
        date = today_str()

    paths = []
    for card in cards:
        path = GAMECARD_DIR / date / f"gamecard_{card['metadata']['game_pk']}.json"
        paths.append(save_json(card, path))

    return paths


def run_gamecard_engine(daily, save=True):
    cards = build_game_cards(daily)

    paths = []
    if save:
        paths = save_game_cards(cards, daily["date"])

    ready = sum(c["readiness"]["pregame_status"] == "READY" for c in cards)
    partial = sum(c["readiness"]["pregame_status"] == "PARTIAL" for c in cards)

    print("=" * 60)
    print("ATLAS GAME CARD ENGINE")
    print("=" * 60)
    print(f"Date.............. {daily['date']}")
    print(f"Game Cards........ {len(cards)}")
    print(f"READY............. {ready}")
    print(f"PARTIAL........... {partial}")
    print("=" * 60)

    return {
        "date": daily["date"],
        "cards": cards,
        "paths": paths,
    }
