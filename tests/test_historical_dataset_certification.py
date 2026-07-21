from __future__ import annotations

import pandas as pd

from atlas.audit.historical_dataset_certification import (
    certify_historical_datasets,
)
from atlas.audit.terminal_score_propagation import (
    repair_terminal_score_propagation,
)


def test_terminal_score_repair_is_additive_and_deterministic():
    master = pd.DataFrame(
        {
            "game_pk": [1],
            "atlas_season": [2024],
            "home_score": [4],
            "away_score": [3],
            "run_differential": [0],
            "preserved_stat": [99],
        }
    )
    team = pd.DataFrame(
        {
            "game_pk": [1, 1],
            "atlas_season": [2024, 2024],
            "home_away": ["home", "away"],
            "runs_scored": [3, 3],
            "runs_allowed": [3, 3],
            "run_differential": [0, 0],
            "won": [False, True],
            "preserved_stat": [10, 20],
        }
    )
    repaired_master, repaired_team, audit = repair_terminal_score_propagation(
        master, team
    )
    assert repaired_master["run_differential"].tolist() == [1]
    assert repaired_master["preserved_stat"].tolist() == [99]
    assert repaired_team["runs_scored"].tolist() == [4, 3]
    assert repaired_team["runs_allowed"].tolist() == [3, 4]
    assert repaired_team["run_differential"].tolist() == [1, -1]
    assert repaired_team["won"].tolist() == [True, False]
    assert repaired_team["preserved_stat"].tolist() == [10, 20]
    assert audit["verification"]["team_pair_winner_errors"] == 0
    assert master["run_differential"].tolist() == [0]
    assert team["runs_scored"].tolist() == [3, 3]


def test_certification_excludes_cancelled_schedule_game():
    schedule = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 1,
                        "gameType": "R",
                        "status": {"detailedState": "Final"},
                    },
                    {
                        "gamePk": 2,
                        "gameType": "R",
                        "status": {"detailedState": "Cancelled"},
                    },
                ]
            }
        ]
    }
    master = pd.DataFrame(
        {
            "game_pk": [1],
            "atlas_season": [2024],
            "game_type": ["R"],
            "home_score": [4],
            "away_score": [3],
            "total_runs": [7],
            "run_differential": [1],
            "home_win": [True],
            "away_win": [False],
        }
    )
    pitch = pd.DataFrame(
        {
            "game_pk": [1],
            "atlas_season": [2024],
            "game_type": ["R"],
            "at_bat_number": [1],
            "pitch_number": [1],
        }
    )
    team = pd.DataFrame(
        {
            "game_pk": [1, 1],
            "atlas_season": [2024, 2024],
            "runs_scored": [4, 3],
            "runs_allowed": [3, 4],
            "run_differential": [1, -1],
            "won": [True, False],
        }
    )
    report = certify_historical_datasets(
        schedule, master, pitch, team, season=2024
    )
    assert report["verdict"] == "certified_with_documented_exceptions"
    assert report["schedule"]["cancelled_game_pks"] == [2]
    assert report["errors"] == []
