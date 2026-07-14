"""
Regression tests for ATLAS Baseball Brain Phase 2B.

Phase 2B converts each verified game-level outcome into exactly
two team-perspective factual outcome rows.

It does not explain outcomes, update identities, discover
concepts, or create predictions.
"""

from pathlib import Path

import pandas as pd

from atlas.game_intelligence.reconstruction import (
    reconstruct_game,
)

from atlas.game_intelligence.outcome_classifier import (
    classify_game_outcome,
)

from atlas.game_intelligence.team_outcome_classifier import (
    classify_team_outcomes,
    team_outcomes_to_frame,
)


REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

TEAM_OUTCOME_PATH = (
    REPO_ROOT
    / "data"
    / "game_intelligence"
    / "team_outcomes"
    / "2024"
    / "team_game_outcomes.parquet"
)

AUDIT_PATH = (
    REPO_ROOT
    / "data"
    / "game_intelligence"
    / "team_outcomes"
    / "2024"
    / "team_game_outcome_audit.parquet"
)

FAILURE_PATH = (
    REPO_ROOT
    / "data"
    / "game_intelligence"
    / "team_outcomes"
    / "2024"
    / "team_game_outcome_failures.parquet"
)


def _classify_team_frame(
    game_pk: int,
) -> pd.DataFrame:
    reconstruction = reconstruct_game(
        game_pk=game_pk,
        season=2024,
    )

    game_outcome = classify_game_outcome(
        reconstruction
    )

    team_outcomes = classify_team_outcomes(
        game_outcome
    )

    return team_outcomes_to_frame(
        team_outcomes
    )


def test_ordinary_shutout_team_perspectives() -> None:
    frame = _classify_team_frame(
        744795
    )

    kc = frame[
        frame["team"].eq("KC")
    ].iloc[0]

    wsh = frame[
        frame["team"].eq("WSH")
    ].iloc[0]

    assert bool(kc["won"])
    assert not bool(kc["lost"])
    assert bool(kc["shutout_win"])
    assert not bool(kc["shutout_loss"])
    assert int(kc["team_score"]) == 3
    assert int(kc["opponent_score"]) == 0
    assert int(kc["run_differential"]) == 3

    assert bool(wsh["lost"])
    assert not bool(wsh["won"])
    assert bool(wsh["shutout_loss"])
    assert not bool(wsh["shutout_win"])
    assert int(wsh["team_score"]) == 0
    assert int(wsh["opponent_score"]) == 3
    assert int(wsh["run_differential"]) == -3


def test_extra_inning_walkoff_team_perspectives() -> None:
    frame = _classify_team_frame(
        745039
    )

    tex = frame[
        frame["team"].eq("TEX")
    ].iloc[0]

    chc = frame[
        frame["team"].eq("CHC")
    ].iloc[0]

    assert bool(tex["walkoff_win"])
    assert not bool(tex["walkoff_loss"])
    assert bool(tex["comeback_win"])
    assert not bool(tex["comeback_loss"])
    assert int(
        tex["largest_deficit_overcome"]
    ) == 1

    assert bool(chc["walkoff_loss"])
    assert not bool(chc["walkoff_win"])
    assert bool(chc["comeback_loss"])
    assert not bool(chc["comeback_win"])
    assert int(
        chc["largest_lead_lost"]
    ) == 1


def test_multi_run_walkoff_team_perspectives() -> None:
    frame = _classify_team_frame(
        746576
    )

    col = frame[
        frame["team"].eq("COL")
    ].iloc[0]

    tb = frame[
        frame["team"].eq("TB")
    ].iloc[0]

    assert bool(col["walkoff_win"])
    assert bool(col["comeback_win"])
    assert int(col["team_score"]) == 10
    assert int(col["opponent_score"]) == 7
    assert int(col["run_differential"]) == 3
    assert int(col["walkoff_runs"]) == 4

    assert bool(tb["walkoff_loss"])
    assert bool(tb["comeback_loss"])
    assert int(tb["team_score"]) == 7
    assert int(tb["opponent_score"]) == 10
    assert int(tb["run_differential"]) == -3
    assert int(tb["walkoff_runs"]) == 4


def test_full_2024_team_outcome_artifact() -> None:
    assert TEAM_OUTCOME_PATH.exists()

    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    assert len(outcomes) == 4856

    assert outcomes[
        "game_pk"
    ].nunique() == 2428

    assert outcomes[
        "team"
    ].nunique() == 30

    assert int(
        outcomes.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    ) == 0

    assert outcomes.groupby(
        "game_pk"
    ).size().eq(2).all()

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


def test_exactly_one_winner_and_loser_per_game() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    assert outcomes.groupby(
        "game_pk"
    )["won"].sum().eq(1).all()

    assert outcomes.groupby(
        "game_pk"
    )["lost"].sum().eq(1).all()

    assert outcomes[
        [
            "won",
            "lost",
        ]
    ].astype("int64").sum(
        axis=1
    ).eq(1).all()


def test_home_away_score_mirroring() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    home = outcomes[
        outcomes["home_away"].eq(
            "HOME"
        )
    ].copy()

    away = outcomes[
        outcomes["home_away"].eq(
            "AWAY"
        )
    ].copy()

    paired = home.merge(
        away,
        on="game_pk",
        suffixes=(
            "_home",
            "_away",
        ),
        validate="one_to_one",
    )

    assert paired[
        "team_home"
    ].eq(
        paired[
            "opponent_away"
        ]
    ).all()

    assert paired[
        "team_away"
    ].eq(
        paired[
            "opponent_home"
        ]
    ).all()

    assert paired[
        "team_score_home"
    ].eq(
        paired[
            "opponent_score_away"
        ]
    ).all()

    assert paired[
        "team_score_away"
    ].eq(
        paired[
            "opponent_score_home"
        ]
    ).all()

    assert paired[
        "run_differential_home"
    ].eq(
        -paired[
            "run_differential_away"
        ]
    ).all()


def test_walkoff_and_comeback_pairing() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    home = outcomes[
        outcomes["home_away"].eq(
            "HOME"
        )
    ].copy()

    away = outcomes[
        outcomes["home_away"].eq(
            "AWAY"
        )
    ].copy()

    paired = home.merge(
        away,
        on="game_pk",
        suffixes=(
            "_home",
            "_away",
        ),
        validate="one_to_one",
    )

    assert paired[
        "walkoff_win_home"
    ].eq(
        paired[
            "walkoff_loss_away"
        ]
    ).all()

    assert not paired[
        "walkoff_win_away"
    ].any()

    assert not paired[
        "walkoff_loss_home"
    ].any()

    assert paired[
        "comeback_win_home"
    ].eq(
        paired[
            "comeback_loss_away"
        ]
    ).all()

    assert paired[
        "comeback_win_away"
    ].eq(
        paired[
            "comeback_loss_home"
        ]
    ).all()


def test_margin_threshold_labels() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    differential = outcomes[
        "run_differential"
    ]

    assert outcomes[
        "won_by_1"
    ].eq(
        outcomes["won"]
        & differential.eq(1)
    ).all()

    assert outcomes[
        "won_by_2_plus"
    ].eq(
        outcomes["won"]
        & differential.ge(2)
    ).all()

    assert outcomes[
        "won_by_4_plus"
    ].eq(
        outcomes["won"]
        & differential.ge(4)
    ).all()

    assert outcomes[
        "won_by_6_plus"
    ].eq(
        outcomes["won"]
        & differential.ge(6)
    ).all()

    assert outcomes[
        "lost_by_1"
    ].eq(
        outcomes["lost"]
        & differential.eq(-1)
    ).all()

    assert outcomes[
        "lost_by_2_plus"
    ].eq(
        outcomes["lost"]
        & differential.le(-2)
    ).all()

    assert outcomes[
        "covered_minus_1_5_result"
    ].eq(
        differential.ge(2)
    ).all()

    assert outcomes[
        "covered_minus_3_5_result"
    ].eq(
        differential.ge(4)
    ).all()

    assert outcomes[
        "covered_minus_5_5_result"
    ].eq(
        differential.ge(6)
    ).all()

    assert outcomes[
        "lost_plus_1_5_result"
    ].eq(
        differential.le(-2)
    ).all()

    assert outcomes[
        "lost_plus_3_5_result"
    ].eq(
        differential.le(-4)
    ).all()

    assert outcomes[
        "lost_plus_5_5_result"
    ].eq(
        differential.le(-6)
    ).all()


def test_scoring_and_allowed_labels() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    scored = outcomes[
        "team_score"
    ]

    allowed = outcomes[
        "opponent_score"
    ]

    assert outcomes[
        "team_scored_0"
    ].eq(
        scored.eq(0)
    ).all()

    assert outcomes[
        "team_scored_3_or_less"
    ].eq(
        scored.le(3)
    ).all()

    assert outcomes[
        "team_scored_exactly_4"
    ].eq(
        scored.eq(4)
    ).all()

    assert outcomes[
        "team_scored_5_plus"
    ].eq(
        scored.ge(5)
    ).all()

    assert outcomes[
        "team_scored_8_plus"
    ].eq(
        scored.ge(8)
    ).all()

    assert outcomes[
        "team_allowed_0"
    ].eq(
        allowed.eq(0)
    ).all()

    assert outcomes[
        "team_allowed_3_or_less"
    ].eq(
        allowed.le(3)
    ).all()

    assert outcomes[
        "team_allowed_exactly_4"
    ].eq(
        allowed.eq(4)
    ).all()

    assert outcomes[
        "team_allowed_5_plus"
    ].eq(
        allowed.ge(5)
    ).all()

    assert outcomes[
        "team_allowed_8_plus"
    ].eq(
        allowed.ge(8)
    ).all()


def test_inning_scoring_mirroring() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    home = outcomes[
        outcomes["home_away"].eq(
            "HOME"
        )
    ].copy()

    away = outcomes[
        outcomes["home_away"].eq(
            "AWAY"
        )
    ].copy()

    paired = home.merge(
        away,
        on="game_pk",
        suffixes=(
            "_home",
            "_away",
        ),
        validate="one_to_one",
    )

    inning_ranges = [
        "innings_1_3",
        "innings_4_6",
        "innings_7_plus",
    ]

    for inning_range in inning_ranges:
        assert paired[
            f"team_runs_{inning_range}_home"
        ].eq(
            paired[
                f"opponent_runs_{inning_range}_away"
            ]
        ).all()

        assert paired[
            f"team_runs_{inning_range}_away"
        ].eq(
            paired[
                f"opponent_runs_{inning_range}_home"
            ]
        ).all()


def test_full_2024_team_outcome_audit() -> None:
    assert AUDIT_PATH.exists()

    audit = pd.read_parquet(
        AUDIT_PATH
    )

    assert len(audit) == 2428

    assert audit[
        "team_outcome_audit_pass"
    ].all()

    assert audit[
        "two_rows"
    ].all()

    assert audit[
        "teams_cross_match"
    ].all()

    assert audit[
        "scores_cross_match"
    ].all()

    assert audit[
        "differentials_are_opposites"
    ].all()

    assert audit[
        "walkoff_perspectives_correct"
    ].all()

    assert audit[
        "comeback_perspectives_correct"
    ].all()

    assert audit[
        "inning_runs_mirror"
    ].all()

    assert audit[
        "provenance_pass"
    ].all()


def test_no_saved_team_outcome_failures() -> None:
    assert FAILURE_PATH.exists()

    failures = pd.read_parquet(
        FAILURE_PATH
    )

    assert failures.empty


def test_known_2024_team_outcome_counts() -> None:
    outcomes = pd.read_parquet(
        TEAM_OUTCOME_PATH
    )

    assert int(
        outcomes["won"].sum()
    ) == 2428

    assert int(
        outcomes["lost"].sum()
    ) == 2428

    assert int(
        outcomes[
            "won_by_1"
        ].sum()
    ) == 675

    assert int(
        outcomes[
            "won_by_2_plus"
        ].sum()
    ) == 1753

    assert int(
        outcomes[
            "won_by_4_plus"
        ].sum()
    ) == 975

    assert int(
        outcomes[
            "won_by_6_plus"
        ].sum()
    ) == 479

    assert int(
        outcomes[
            "shutout_win"
        ].sum()
    ) == 321

    assert int(
        outcomes[
            "shutout_loss"
        ].sum()
    ) == 321

    assert int(
        outcomes[
            "walkoff_win"
        ].sum()
    ) == 208

    assert int(
        outcomes[
            "walkoff_loss"
        ].sum()
    ) == 208

    assert int(
        outcomes[
            "comeback_win"
        ].sum()
    ) == 1026

    assert int(
        outcomes[
            "comeback_loss"
        ].sum()
    ) == 1026

    assert int(
        outcomes[
            "team_scored_0"
        ].sum()
    ) == 321

    assert int(
        outcomes[
            "team_scored_5_plus"
        ].sum()
    ) == 2054
