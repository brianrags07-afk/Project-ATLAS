
from atlas.profiles.profile_utils import blank_profile, safe_rate, confidence_from_sample


def summarize_team_games(df):
    games = len(df)
    wins = int(df["won"].sum()) if "won" in df else 0
    runs = df["runs_scored"].sum()
    allowed = df["runs_allowed"].sum()

    return {
        "games": int(games),
        "wins": wins,
        "losses": int(games - wins),
        "win_pct": safe_rate(wins, games),
        "runs_per_game": safe_rate(runs, games),
        "runs_allowed_per_game": safe_rate(allowed, games),
        "avg_run_differential": float(df["run_differential"].mean()) if games else None,
    }


def build_split_contexts(team_df):
    contexts = {}

    if "home_away" in team_df.columns:
        for label, split_df in team_df.groupby("home_away"):
            contexts[str(label).lower()] = summarize_team_games(split_df)

    return contexts


def build_team_offense_profile(team_df):
    profile = blank_profile()

    games = len(team_df)
    runs = team_df["runs_scored"].sum()
    hits = team_df["hits"].sum()
    walks = team_df["walks"].sum()
    strikeouts = team_df["strikeouts"].sum()
    at_bats = team_df["at_bats"].sum()

    singles = team_df["singles"].sum()
    doubles = team_df["doubles"].sum()
    triples = team_df["triples"].sum()
    home_runs = team_df["home_runs"].sum()
    total_bases = singles + 2*doubles + 3*triples + 4*home_runs

    profile["questions"] = [
        "When does this team score?",
        "Why does this team score?",
        "Against who does this offense become dangerous?",
        "Under what conditions does this offense create overs?"
    ]

    profile["facts"] = {
        "games": int(games),
        "runs": int(runs),
        "runs_per_game": safe_rate(runs, games),
        "hits": int(hits),
        "hits_per_game": safe_rate(hits, games),
        "walks": int(walks),
        "walks_per_game": safe_rate(walks, games),
        "strikeouts": int(strikeouts),
        "strikeouts_per_game": safe_rate(strikeouts, games),
        "batting_average": safe_rate(hits, at_bats),
        "home_runs": int(home_runs),
        "home_runs_per_game": safe_rate(home_runs, games),
        "total_bases": int(total_bases),
        "total_bases_per_game": safe_rate(total_bases, games),
    }

    profile["contexts"] = build_split_contexts(team_df)
    profile["samples"] = {"games": int(games)}
    profile["confidence"] = {"sample_strength": confidence_from_sample(games)}
    profile["evidence"] = {"source": "team_game_state"}

    return profile


def build_team_outcome_profiles(team_df):
    moneyline = blank_profile()
    totals = blank_profile()
    runline = blank_profile()

    games = len(team_df)
    wins = int(team_df["won"].sum())
    total_runs = team_df["runs_scored"] + team_df["runs_allowed"]

    moneyline["questions"] = [
        "When does this team win?",
        "Why does this team win?",
        "Against who does this team win?",
        "Under what conditions does this team lose?"
    ]

    moneyline["facts"] = {
        "games": int(games),
        "wins": wins,
        "losses": int(games - wins),
        "win_pct": safe_rate(wins, games),
        "avg_run_differential": float(team_df["run_differential"].mean()) if games else None,
    }

    totals["questions"] = [
        "When do this team's games go over?",
        "When do this team's games stay under?",
        "Which contexts create higher or lower run environments?"
    ]

    totals["facts"] = {
        "games": int(games),
        "avg_total_runs": float(total_runs.mean()) if games else None,
        "median_total_runs": float(total_runs.median()) if games else None,
        "games_9_plus_runs": int((total_runs >= 9).sum()),
        "games_7_or_less_runs": int((total_runs <= 7).sum()),
    }

    runline["questions"] = [
        "When does this team win comfortably?",
        "When does this team play close games?",
        "When does this team get blown out?"
    ]

    runline["facts"] = {
        "games": int(games),
        "avg_margin": float(team_df["run_differential"].mean()) if games else None,
        "one_run_games": int((team_df["run_differential"].abs() == 1).sum()),
        "wins_by_2_plus": int(((team_df["won"] == True) & (team_df["run_differential"] >= 2)).sum()),
        "losses_by_2_plus": int(((team_df["won"] == False) & (team_df["run_differential"] <= -2)).sum()),
    }

    for profile in [moneyline, totals, runline]:
        profile["contexts"] = build_split_contexts(team_df)
        profile["samples"] = {"games": int(games)}
        profile["confidence"] = {"sample_strength": confidence_from_sample(games)}
        profile["evidence"] = {"source": "team_game_state"}

    return moneyline, totals, runline


def attach_team_profiles(card, team_df):
    card["offense_profile"] = build_team_offense_profile(team_df)

    moneyline, totals, runline = build_team_outcome_profiles(team_df)
    card["moneyline_profile"] = moneyline
    card["totals_profile"] = totals
    card["runline_profile"] = runline

    return card
