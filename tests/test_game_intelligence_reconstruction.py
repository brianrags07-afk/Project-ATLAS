"""
Regression tests for ATLAS Baseball Brain Phase 1.

Phase 1 is frozen after these tests pass.
"""

from pathlib import Path

import pandas as pd

from atlas.game_intelligence.reconstruction import (
    reconstruct_game,
)
from atlas.game_intelligence.contracts import (
    RECONSTRUCTION_AUDIT_DIR,
)


SEASON = 2024


def _assert_standard_game(
    game_pk: int,
    expected_home_score: int,
    expected_away_score: int,
) -> None:
    game = reconstruct_game(
        game_pk=game_pk,
        season=SEASON,
    )

    assert len(game.game_core) == 1
    assert len(game.manifest) == 1
    assert len(game.pregame_teams) == 2
    assert len(game.lineups) == 2
    assert len(game.bullpens) == 2
    assert len(game.events) > 0
    assert len(game.targets) == 1

    assert game.validation[
        "scores_agree"
    ]

    assert game.validation[
        "all_available_pregame_safety_checks_pass"
    ]

    assert game.validation[
        "reconstruction_pass"
    ]

    core = game.game_core.iloc[0]
    target = game.targets.iloc[0]

    assert int(
        core["home_score"]
    ) == expected_home_score

    assert int(
        core["away_score"]
    ) == expected_away_score

    assert int(
        target["home_score"]
    ) == expected_home_score

    assert int(
        target["away_score"]
    ) == expected_away_score

    event_home_score = int(
        pd.to_numeric(
            game.events[
                "post_home_score"
            ],
            errors="raise",
        ).iloc[-1]
    )

    event_away_score = int(
        pd.to_numeric(
            game.events[
                "post_away_score"
            ],
            errors="raise",
        ).iloc[-1]
    )

    assert (
        event_home_score
        == expected_home_score
    )

    assert (
        event_away_score
        == expected_away_score
    )


def test_ordinary_completed_game() -> None:
    _assert_standard_game(
        game_pk=744795,
        expected_home_score=0,
        expected_away_score=3,
    )


def test_one_run_walkoff_repair() -> None:
    _assert_standard_game(
        game_pk=745039,
        expected_home_score=4,
        expected_away_score=3,
    )


def test_multi_run_walkoff_repair() -> None:
    _assert_standard_game(
        game_pk=746576,
        expected_home_score=10,
        expected_away_score=7,
    )


def test_terminal_replay_score_repair() -> None:
    _assert_standard_game(
        game_pk=746137,
        expected_home_score=2,
        expected_away_score=3,
    )


def test_known_source_exception() -> None:
    game = reconstruct_game(
        game_pk=746942,
        season=SEASON,
    )

    assert len(game.game_core) == 1
    assert len(game.manifest) == 1
    assert len(game.events) > 0
    assert len(game.lineups) == 2

    assert len(
        game.pregame_teams
    ) == 0

    assert len(
        game.bullpens
    ) == 0

    assert len(
        game.targets
    ) == 0

    assert game.validation[
        "known_exception"
    ]

    assert not game.validation[
        "reconstruction_pass"
    ]


def test_full_2024_audit_artifact() -> None:
    audit_path = (
        RECONSTRUCTION_AUDIT_DIR
        / "2024"
        / "season_reconstruction_audit.parquet"
    )

    assert audit_path.exists()

    audit = pd.read_parquet(
        audit_path
    )

    assert len(audit) == 2429
    assert audit["game_pk"].nunique() == 2429
    assert int(
        audit["game_pk"]
        .duplicated()
        .sum()
    ) == 0

    counts = (
        audit["audit_status"]
        .value_counts()
    )

    assert int(
        counts.get(
            "pass",
            0,
        )
    ) == 2428

    assert int(
        counts.get(
            "known_exception",
            0,
        )
    ) == 1

    assert int(
        counts.get(
            "fail",
            0,
        )
    ) == 0

    assert int(
        (~audit["scores_agree"])
        .sum()
    ) == 0

    exception = audit[
        audit["audit_status"].eq(
            "known_exception"
        )
    ]

    assert len(exception) == 1

    assert int(
        exception[
            "game_pk"
        ].iloc[0]
    ) == 746942
