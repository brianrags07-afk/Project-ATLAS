
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


def stability_from_games(games):
    if games >= 300:
        return "high"
    if games >= 150:
        return "medium"
    return "low"
