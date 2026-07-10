
def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def absolute_change(first, second):
    first = safe_float(first)
    second = safe_float(second)

    if first is None or second is None:
        return None

    return second - first


def classify_direction(change, tolerance=0.01):
    if change is None:
        return "unknown"

    if change > tolerance:
        return "improved"

    if change < -tolerance:
        return "declined"

    return "stable"


def strongest_season_shift(season_breakdown):
    seasons = sorted(season_breakdown.keys())

    if len(seasons) < 2:
        return None

    strongest = None

    for previous, current in zip(seasons[:-1], seasons[1:]):
        previous_rate = safe_float(
            season_breakdown[previous].get("win_pct")
        )
        current_rate = safe_float(
            season_breakdown[current].get("win_pct")
        )

        if previous_rate is None or current_rate is None:
            continue

        change = current_rate - previous_rate

        record = {
            "from_season": previous,
            "to_season": current,
            "win_pct_change": change,
            "absolute_change": abs(change),
            "direction": classify_direction(change),
        }

        if strongest is None or record["absolute_change"] > strongest["absolute_change"]:
            strongest = record

    return strongest
