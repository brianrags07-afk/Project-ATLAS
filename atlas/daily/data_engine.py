import pandas as pd

from atlas.config import MLB_API, today_str
from atlas.utils.api import get_json, safe_get


def pull_schedule(date=None):
    if date is None:
        date = today_str()

    return get_json(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "startDate": date,
            "endDate": date,
            "hydrate": "team,probablePitcher,venue,status",
        },
    )


def extract_games(schedule):
    return safe_get(schedule, ["dates", 0, "games"], [])


def build_starters(games):
    rows = []

    for game in games:
        game_pk = game["gamePk"]

        for side, home_away in [("away", "Away"), ("home", "Home")]:
            team = game["teams"][side]["team"]
            pitcher = game["teams"][side].get("probablePitcher")

            rows.append({
                "game_pk": game_pk,
                "team": team.get("name"),
                "team_id": team.get("id"),
                "home_away": home_away,
                "pitcher": pitcher.get("fullName") if pitcher else "TBD",
                "pitcher_id": pitcher.get("id") if pitcher else None,
                "starter_ready": pitcher is not None,
            })

    return pd.DataFrame(rows)


def pull_live_feed(game_pk):
    return get_json(f"{MLB_API}.1/game/{game_pk}/feed/live")


def build_lineups(games):
    rows = []
    live_feeds = {}

    for game in games:
        game_pk = game["gamePk"]

        try:
            feed = pull_live_feed(game_pk)
            live_feeds[game_pk] = feed
        except Exception as e:
            print(f"FAILED live feed {game_pk}: {e}")
            continue

        players = feed.get("gameData", {}).get("players", {})
        boxscore = safe_get(feed, ["liveData", "boxscore", "teams"], {})

        for side, home_away in [("away", "Away"), ("home", "Home")]:
            team_name = game["teams"][side]["team"]["name"]
            batting_order = safe_get(boxscore, [side, "battingOrder"], []) or []

            for order, player_id in enumerate(batting_order, start=1):
                player = players.get(f"ID{player_id}", {})

                rows.append({
                    "game_pk": game_pk,
                    "team": team_name,
                    "home_away": home_away,
                    "batting_order": order,
                    "player_name": player.get("fullName"),
                    "player_id": player_id,
                    "bat_side": safe_get(player, ["batSide", "code"]),
                    "throw_side": safe_get(player, ["pitchHand", "code"]),
                    "position": safe_get(player, ["primaryPosition", "abbreviation"]),
                })

    return pd.DataFrame(rows), live_feeds


def build_readiness(games, starters, lineups):
    rows = []

    for game in games:
        game_pk = game["gamePk"]

        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]

        away_starter_ready = bool(
            starters[
                (starters["game_pk"] == game_pk)
                & (starters["home_away"] == "Away")
            ]["starter_ready"].iloc[0]
        )

        home_starter_ready = bool(
            starters[
                (starters["game_pk"] == game_pk)
                & (starters["home_away"] == "Home")
            ]["starter_ready"].iloc[0]
        )

        away_lineup_count = len(
            lineups[
                (lineups["game_pk"] == game_pk)
                & (lineups["team"] == away_team)
            ]
        )

        home_lineup_count = len(
            lineups[
                (lineups["game_pk"] == game_pk)
                & (lineups["team"] == home_team)
            ]
        )

        ready = (
            away_starter_ready
            and home_starter_ready
            and away_lineup_count >= 9
            and home_lineup_count >= 9
        )

        rows.append({
            "game_pk": game_pk,
            "away_team": away_team,
            "home_team": home_team,
            "away_starter_ready": away_starter_ready,
            "home_starter_ready": home_starter_ready,
            "away_lineup_count": away_lineup_count,
            "home_lineup_count": home_lineup_count,
            "pregame_status": "READY" if ready else "PARTIAL",
        })

    return pd.DataFrame(rows)


def run_daily_engine(date=None):
    if date is None:
        date = today_str()

    schedule = pull_schedule(date)
    games = extract_games(schedule)
    starters = build_starters(games)
    lineups, live_feeds = build_lineups(games)
    readiness = build_readiness(games, starters, lineups)

    daily = {
        "date": date,
        "schedule": schedule,
        "games": games,
        "starters": starters,
        "lineups": lineups,
        "live_feeds": live_feeds,
        "readiness": readiness,
    }

    print("=" * 60)
    print("ATLAS DAILY ENGINE")
    print("=" * 60)
    print(f"Date.............. {date}")
    print(f"Games............. {len(games)}")
    print(f"Starters.......... {len(starters)}")
    print(f"Lineups........... {len(lineups)}")
    print(f"Live Feeds........ {len(live_feeds)}")

    ready = (readiness["pregame_status"] == "READY").sum()
    partial = (readiness["pregame_status"] == "PARTIAL").sum()

    print(f"READY Games....... {ready}")
    print(f"PARTIAL Games..... {partial}")
    print("=" * 60)

    return daily
