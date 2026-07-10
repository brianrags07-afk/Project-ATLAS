
import math


def safe_rate(numerator, denominator):
    if denominator in [0, None]:
        return None
    return numerator / denominator


def confidence_from_games(games):
    if games >= 300:
        return "very_strong"
    if games >= 150:
        return "strong"
    if games >= 75:
        return "moderate"
    if games >= 30:
        return "limited"
    return "weak"


def stability_from_season_rates(season_rates):
    valid_rates = [
        float(rate)
        for rate in season_rates.values()
        if rate is not None
    ]

    if len(valid_rates) < 2:
        return {
            "label": "not_measured",
            "range": None,
            "std_dev": None,
            "seasons_measured": len(valid_rates),
        }

    rate_range = max(valid_rates) - min(valid_rates)

    mean_rate = sum(valid_rates) / len(valid_rates)
    variance = sum(
        (rate - mean_rate) ** 2
        for rate in valid_rates
    ) / len(valid_rates)
    std_dev = math.sqrt(variance)

    if rate_range <= 0.05:
        label = "high"
    elif rate_range <= 0.10:
        label = "medium"
    else:
        label = "low"

    return {
        "label": label,
        "range": rate_range,
        "std_dev": std_dev,
        "seasons_measured": len(valid_rates),
    }


def missing_percentage(df, required_columns):
    if df.empty or not required_columns:
        return None

    total_cells = len(df) * len(required_columns)
    if total_cells == 0:
        return None

    missing_cells = int(df[required_columns].isna().sum().sum())
    return missing_cells / total_cells
