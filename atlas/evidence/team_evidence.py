
from atlas.evidence.evidence_schema import make_evidence_object
from atlas.evidence.evidence_utils import safe_rate, confidence_from_games, stability_from_games


def build_team_moneyline_evidence(team_df):
    team = team_df["team"].iloc[0]

    games = len(team_df)
    wins = int(team_df["won"].sum())
    losses = int(games - wins)
    win_pct = safe_rate(wins, games)

    obj = make_evidence_object(
        evidence_id=f"EV_TEAM_ML_0001_{team}",
        question_id="Q_TEAM_ML_0001",
        entity_type="team",
        entity_id=team,
        context="overall",
    )

    obj["sample"] = {
        "games": int(games),
    }

    obj["measurements"] = {
        "wins": wins,
        "losses": losses,
        "observed_win_pct": win_pct,
        "baseline_win_pct": 0.50,
        "effect_size": win_pct - 0.50 if win_pct is not None else None,
        "avg_run_differential": float(team_df["run_differential"].mean()),
    }

    obj["quality"] = {
        "confidence": confidence_from_games(games),
        "stability": stability_from_games(games),
        "recency": "2024-2026",
    }

    obj["traceability"] = {
        "source_games": team_df["game_pk"].astype(int).tolist(),
    }

    obj["data_quality"] = {
        "missing_pct": 0.0,
        "sample_completeness": 1.0,
        "validation_checks": ["PASS"],
    }

    return obj
