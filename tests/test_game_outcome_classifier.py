"""
Regression tests for ATLAS Baseball Brain Phase 2A.

Phase 2A provides deterministic factual game-outcome
classification. It does not explain outcomes, update identities,
or create predictions.
"""

from pathlib import Path

import pandas as pd

from atlas.game_intelligence.reconstruction import (
    reconstruct_game,
)

from atlas.game_intelligence.outcome_classifier import (
    classify_game_outcome,
)


REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

OUTCOME_PATH = (
    REPO_ROOT
    / "data"
    / "game_intelligence"
    / "outcomes"
    / "2024"
    / "game_outcomes.parquet"
)

AUDIT_PATH = (
    REPO_ROOT
    / "data"
    / "game_intelligence"
    / "outcomes"
    / "2024"
    / "game_outcome_audit.parquet"
)

FAILURE_PATH = (
    REPO_ROOT
    / "data"
    / "game_intelligence"
    / "outcomes"
    / "2024"
    / "game_outcome_failures.parquet"
)


def _classify(
    game_pk: int,
):
    reconstruction = reconstruct_game(
        game_pk=game_pk,
        season=2024,
    )

    return classify_game_outcome(
        reconstruction
    )


def test_ordinary_road_shutout() -> None:
    outcome = _classify(
        744795
    )

    assert outcome.home_team == "WSH"
    assert outcome.away_team == "KC"

    assert outcome.home_score == 0
    assert outcome.away_score == 3

    assert outcome.winner_side == "AWAY"
    assert outcome.winner_team == "KC"
    assert outcome.loser_team == "WSH"

    assert outcome.away_win
    assert not outcome.home_win

    assert outcome.away_shutout_win
    assert outcome.either_team_shut_out

    assert outcome.total_runs == 3
    assert outcome.absolute_run_margin == 3

    assert not outcome.walkoff
    assert not outcome.extra_innings
    assert not outcome.comeback_win


def test_extra_inning_walkoff() -> None:
    outcome = _classify(
        745039
    )

    assert outcome.home_team == "TEX"
    assert outcome.away_team == "CHC"

    assert outcome.home_score == 4
    assert outcome.away_score == 3

    assert outcome.home_win
    assert outcome.one_run_game

    assert outcome.extra_innings
    assert outcome.tied_after_regulation

    assert outcome.walkoff
    assert outcome.walkoff_runs == 1

    assert outcome.home_comeback_win
    assert outcome.comeback_win

    assert outcome.lead_changes == 1


def test_multi_run_walkoff() -> None:
    outcome = _classify(
        746576
    )

    assert outcome.home_team == "COL"
    assert outcome.away_team == "TB"

    assert outcome.home_score == 10
    assert outcome.away_score == 7

    assert outcome.walkoff
    assert outcome.walkoff_runs == 4
    assert outcome.terminal_scoring_play

    assert outcome.home_comeback_win
    assert outcome.comeback_win

    assert outcome.absolute_run_margin == 3
    assert outcome.total_runs == 17


def test_terminal_replay_game_not_walkoff() -> None:
    outcome = _classify(
        746137
    )

    assert outcome.home_team == "LAD"
    assert outcome.away_team == "TEX"

    assert outcome.home_score == 2
    assert outcome.away_score == 3

    assert outcome.away_win
    assert outcome.winner_team == "TEX"

    assert not outcome.walkoff
    assert outcome.walkoff_runs == 0

    assert outcome.final_inning_scoring
    assert outcome.comeback_win
    assert outcome.lead_changes == 1


def test_margin_threshold_labels() -> None:
    outcomes = pd.read_parquet(
        OUTCOME_PATH
    )

    absolute_margin = outcomes[
        "absolute_run_margin"
    ]

    assert outcomes[
        "one_run_game"
    ].eq(
        absolute_margin.eq(1)
    ).all()

    assert outcomes[
        "margin_gt_1_5"
    ].eq(
        absolute_margin.ge(2)
    ).all()

    assert outcomes[
        "margin_gt_3_5"
    ].eq(
        absolute_margin.ge(4)
    ).all()

    assert outcomes[
        "margin_gt_5_5"
    ].eq(
        absolute_margin.ge(6)
    ).all()


def test_total_threshold_labels() -> None:
    outcomes = pd.read_parquet(
        OUTCOME_PATH
    )

    total = outcomes[
        "total_runs"
    ]

    threshold_map = {
        "game_total_5_or_less":
            total.le(5),

        "game_total_6_or_less":
            total.le(6),

        "game_total_7_or_less":
            total.le(7),

        "game_total_8_or_less":
            total.le(8),

        "game_total_9_plus":
            total.ge(9),

        "game_total_10_plus":
            total.ge(10),

        "game_total_11_plus":
            total.ge(11),

        "game_total_12_plus":
            total.ge(12),

        "game_total_15_plus":
            total.ge(15),

        "game_total_17_plus":
            total.ge(17),
    }

    for column, expected in (
        threshold_map.items()
    ):
        assert outcomes[
            column
        ].eq(
            expected
        ).all()


def test_full_2024_outcome_artifact() -> None:
    assert OUTCOME_PATH.exists()

    outcomes = pd.read_parquet(
        OUTCOME_PATH
    )

    assert len(outcomes) == 2428

    assert outcomes[
        "game_pk"
    ].nunique() == 2428

    assert int(
        outcomes[
            "game_pk"
        ].duplicated().sum()
    ) == 0

    assert 746942 not in set(
        outcomes[
            "game_pk"
        ].astype(int)
    )

    assert outcomes[
        "atlas_season"
    ].eq(2024).all()

    assert outcomes[
        "score_sources_verified"
    ].all()

    assert outcomes[
        "reconstruction_verified"
    ].all()

    assert (
        ~outcomes[
            "prediction_created"
        ]
    ).all()

    assert (
        ~outcomes[
            "identity_updated"
        ]
    ).all()

    assert (
        ~outcomes[
            "explanation_created"
        ]
    ).all()

    assert (
        ~outcomes[
            "future_games_used"
        ]
    ).all()


def test_full_2024_outcome_audit() -> None:
    assert AUDIT_PATH.exists()

    audit = pd.read_parquet(
        AUDIT_PATH
    )

    assert len(audit) == 2428

    assert audit[
        "outcome_audit_pass"
    ].all()

    assert int(
        audit[
            "game_pk"
        ].duplicated().sum()
    ) == 0

    assert audit[
        "final_score_not_tied"
    ].all()

    assert audit[
        "exactly_one_winner"
    ].all()

    assert audit[
        "winner_side_matches_score"
    ].all()

    assert audit[
        "winner_team_matches_score"
    ].all()

    assert audit[
        "loser_team_matches_score"
    ].all()

    assert audit[
        "walkoff_consistent"
    ].all()

    assert audit[
        "comeback_consistent"
    ].all()

    assert audit[
        "provenance_pass"
    ].all()


def test_no_saved_failures() -> None:
    assert FAILURE_PATH.exists()

    failures = pd.read_parquet(
        FAILURE_PATH
    )

    assert failures.empty


def test_comeback_lead_change_equivalence() -> None:
    outcomes = pd.read_parquet(
        OUTCOME_PATH
    )

    assert outcomes[
        "comeback_win"
    ].eq(
        outcomes[
            "lead_changes"
        ].gt(0)
    ).all()

    assert int(
        outcomes[
            "comeback_win"
        ].sum()
    ) == 1026

    assert int(
        outcomes[
            "lead_changes"
        ].gt(0).sum()
    ) == 1026


def test_known_2024_distribution_counts() -> None:
    outcomes = pd.read_parquet(
        OUTCOME_PATH
    )

    assert int(
        outcomes[
            "home_win"
        ].sum()
    ) == 1267

    assert int(
        outcomes[
            "away_win"
        ].sum()
    ) == 1161

    assert int(
        outcomes[
            "one_run_game"
        ].sum()
    ) == 675

    assert int(
        outcomes[
            "either_team_shut_out"
        ].sum()
    ) == 321

    assert int(
        outcomes[
            "extra_innings"
        ].sum()
    ) == 216

    assert int(
        outcomes[
            "walkoff"
        ].sum()
    ) == 208
