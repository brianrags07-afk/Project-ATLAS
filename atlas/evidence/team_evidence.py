
from atlas.evidence.evidence_schema import make_evidence_object
from atlas.evidence.evidence_utils import (
    safe_rate,
    confidence_from_games,
    stability_from_season_rates,
    missing_percentage,
)


REQUIRED_TEAM_ML_COLUMNS = [
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "won",
    "run_differential",
]


def build_team_moneyline_evidence(team_df):
    if team_df.empty:
        raise ValueError("team_df is empty")

    missing_columns = [
        column
        for column in REQUIRED_TEAM_ML_COLUMNS
        if column not in team_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required team evidence columns: {missing_columns}"
        )

    team_df = team_df.copy()
    team_df["game_date"] = team_df["game_date"].astype(str)

    team = str(team_df["team"].iloc[0])

    games = int(len(team_df))
    wins = int(team_df["won"].fillna(False).astype(bool).sum())
    losses = int(games - wins)
    win_pct = safe_rate(wins, games)

    season_breakdown = {}
    season_rates = {}

    for season, season_df in team_df.groupby("atlas_season", dropna=True):
        season_games = int(len(season_df))
        season_wins = int(
            season_df["won"].fillna(False).astype(bool).sum()
        )
        season_losses = int(season_games - season_wins)
        season_win_pct = safe_rate(season_wins, season_games)

        season_key = str(int(season))

        season_breakdown[season_key] = {
            "games": season_games,
            "wins": season_wins,
            "losses": season_losses,
            "win_pct": season_win_pct,
        }

        season_rates[season_key] = season_win_pct

    stability = stability_from_season_rates(season_rates)

    missing_pct = missing_percentage(
        team_df,
        REQUIRED_TEAM_ML_COLUMNS,
    )

    source_games = (
        team_df["game_pk"]
        .dropna()
        .astype(int)
        .tolist()
    )

    obj = make_evidence_object(
        evidence_id=f"EV_TEAM_ML_0001_{team}",
        question_id="Q_TEAM_ML_0001",
        entity_type="team",
        entity_id=team,
        context="overall",
    )

    obj["sample"] = {
        "games": games,
        "seasons": sorted(
            int(season)
            for season in team_df["atlas_season"].dropna().unique()
        ),
        "date_start": team_df["game_date"].min(),
        "date_end": team_df["game_date"].max(),
    }

    obj["measurements"] = {
        "wins": wins,
        "losses": losses,
        "observed_win_pct": win_pct,
        "baseline_win_pct": 0.50,
        "effect_size": (
            win_pct - 0.50
            if win_pct is not None
            else None
        ),
        "avg_run_differential": float(
            team_df["run_differential"].mean()
        ),
        "season_breakdown": season_breakdown,
    }

    obj["quality"] = {
        "confidence": confidence_from_games(games),
        "stability": stability,
        "recency": {
            "date_start": team_df["game_date"].min(),
            "date_end": team_df["game_date"].max(),
            "latest_season": int(
                team_df["atlas_season"].dropna().max()
            ),
        },
    }

    obj["traceability"] = {
        "source_games": source_games,
        "source_game_count": len(source_games),
        "source_dataset": "team_game_state.parquet",
    }

    validation_checks = []

    if wins + losses == games:
        validation_checks.append("WINS_PLUS_LOSSES_EQUALS_GAMES")
    else:
        validation_checks.append("FAIL_WINS_PLUS_LOSSES")

    if len(source_games) == games:
        validation_checks.append("TRACEABILITY_COMPLETE")
    else:
        validation_checks.append("TRACEABILITY_INCOMPLETE")

    if missing_pct == 0:
        validation_checks.append("NO_REQUIRED_FIELD_MISSINGNESS")
    else:
        validation_checks.append("REQUIRED_FIELD_MISSINGNESS_PRESENT")

    obj["data_quality"] = {
        "missing_pct": missing_pct,
        "sample_completeness": safe_rate(
            len(source_games),
            games,
        ),
        "validation_checks": validation_checks,
    }

    return obj
