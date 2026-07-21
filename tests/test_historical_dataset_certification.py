from __future__ import annotations

import pandas as pd

from atlas.audit.historical_dataset_certification import (
    attach_certification_provenance,
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


def test_certification_provenance_is_self_contained():
    identity = {
        "gcs_uri": "gs://bucket/object",
        "generation": "123",
        "sha256": "a" * 64,
    }
    provenance = {
        "schema_version": "1",
        "certified_at_utc": "2026-07-21T14:00:00+00:00",
        "github": {
            "repository": "owner/repo",
            "commit_sha": "abc",
            "run_id": "1",
            "workflow": "certify",
            "ref": "refs/heads/main",
        },
        "transfer_manifest": dict(identity),
        "inputs": {
            "schedule": dict(identity),
            "master": dict(identity),
            "pitch": dict(identity),
            "team_state": dict(identity),
        },
    }
    report = attach_certification_provenance(
        {"verdict": "certified", "errors": []},
        provenance,
    )
    assert report["provenance"]["inputs"]["pitch"]["generation"] == "123"
    assert report["provenance"]["github"]["commit_sha"] == "abc"


def test_certification_accepts_canonical_schedule_rows():
    schedule = [
        {
            "game_pk": 1,
            "season": 2025,
            "game_type_code": "R",
            "detailed_state": "Final",
            "game_state_category": "final",
            "counted_in_expected_games": True,
        },
        {
            "game_pk": 2,
            "season": 2025,
            "game_type_code": "R",
            "detailed_state": "Cancelled",
            "game_state_category": "cancelled",
            "counted_in_expected_games": False,
        },
        {
            "game_pk": 3,
            "season": 2024,
            "game_type_code": "R",
            "detailed_state": "Final",
            "game_state_category": "final",
            "counted_in_expected_games": True,
        },
    ]
    master = pd.DataFrame(
        {
            "game_pk": [1],
            "atlas_season": [2025],
            "game_type": ["R"],
            "home_score": [5],
            "away_score": [3],
            "total_runs": [8],
            "run_differential": [2],
            "home_win": [True],
            "away_win": [False],
        }
    )
    pitch = pd.DataFrame(
        {
            "game_pk": [1],
            "atlas_season": [2025],
            "game_type": ["R"],
            "at_bat_number": [1],
            "pitch_number": [1],
        }
    )
    team = pd.DataFrame(
        {
            "game_pk": [1, 1],
            "atlas_season": [2025, 2025],
            "runs_scored": [5, 3],
            "runs_allowed": [3, 5],
            "run_differential": [2, -2],
            "won": [True, False],
        }
    )

    report = certify_historical_datasets(
        schedule, master, pitch, team, season=2025
    )

    assert report["verdict"] == "certified_with_documented_exceptions"
    assert report["schedule"]["published_regular_games"] == 2
    assert report["schedule"]["completed_games"] == 1
    assert report["schedule"]["cancelled_game_pks"] == [2]
    assert report["errors"] == []
