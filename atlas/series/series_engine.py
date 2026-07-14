from collections import defaultdict

SERIES_ENGINE_VERSION = "1.0.0"


def build_series_intelligence(daily):
    """
    Build basic series context for every game in today's slate.
    """
    series = {}
    grouped = defaultdict(list)

    for game in daily["games"]:
        away = game["teams"]["away"]["team"]["name"]
        home = game["teams"]["home"]["team"]["name"]

        key = tuple(sorted([away, home]))
        grouped[key].append(game)

    for key, games in grouped.items():
        games = sorted(games, key=lambda g: g["gameDate"])
        series_length = len(games)

        for i, game in enumerate(games):
            game_pk = game["gamePk"]

            series[game_pk] = {
                "series_engine_version": SERIES_ENGINE_VERSION,
                "series_id": f"{key[0]}_{key[1]}_{daily['date']}",
                "game_number": i + 1,
                "series_length": series_length,
                "rubber_match": series_length == 3 and i == 2,
                "sweep_opportunity": False,
                "getaway_day": i == series_length - 1,
                "current_series_record": {
                    "away": 0,
                    "home": 0
                }
            }

    return series
