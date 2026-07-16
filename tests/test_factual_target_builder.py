from pathlib import Path

import pandas as pd

from atlas.learning.factual_target_builder import (
    build_game_targets,
    build_team_game_targets,
    standardize_completed_games,
    validate_target_symmetry,
)


ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

MASTER_GAME_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/master/master_game_database.parquet'
)

EVIDENCE_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence.parquet'
)

OUTPUT_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_team_game_learning_targets.parquet'
)

EXCLUDED_GAMES_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/completed_games_excluded_from_evidence_universe.csv'
)


def build_aligned_targets():
    master = pd.read_parquet(
        MASTER_GAME_PATH
    )

    completed, _ = standardize_completed_games(
        master,
        season=2024,
    )

    evidence = pd.read_parquet(
        EVIDENCE_PATH,
        columns=[
            "game_pk",
            "team",
        ],
    )

    approved_game_pks = set(
        evidence[
            "game_pk"
        ].astype(
            "int64"
        )
    )

    aligned = completed[
        completed[
            "game_pk"
        ].isin(
            approved_game_pks
        )
    ].copy()

    games = build_game_targets(
        aligned
    )

    teams = build_team_game_targets(
        games
    )

    return (
        completed,
        evidence,
        games,
        teams,
    )


def test_master_source_has_documented_extra_game():
    completed, evidence, _, _ = build_aligned_targets()

    assert completed[
        "game_pk"
    ].nunique() >= evidence[
        "game_pk"
    ].nunique()

    excluded = set(
        completed[
            "game_pk"
        ]
    ).difference(
        set(
            evidence[
                "game_pk"
            ]
        )
    )

    saved_excluded = pd.read_csv(
        EXCLUDED_GAMES_PATH
    )

    assert len(
        saved_excluded
    ) == len(
        excluded
    )


def test_targets_match_evidence_universe():
    _, evidence, games, teams = build_aligned_targets()

    assert len(
        games
    ) == evidence[
        "game_pk"
    ].nunique()

    assert len(
        teams
    ) == len(
        evidence
    )


def test_duplicate_team_games_zero():
    _, _, _, teams = build_aligned_targets()

    assert teams.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).sum() == 0


def test_exactly_two_rows_per_game():
    _, _, _, teams = build_aligned_targets()

    assert teams.groupby(
        "game_pk"
    ).size().eq(
        2
    ).all()


def test_win_loss_symmetry():
    _, _, _, teams = build_aligned_targets()

    assert teams[
        "target_team_win"
    ].sum() == teams[
        "target_team_loss"
    ].sum()

    assert teams[
        "target_team_win_by_2_plus"
    ].sum() == teams[
        "target_team_loss_by_2_plus"
    ].sum()

    assert teams[
        "target_team_win_by_4_plus"
    ].sum() == teams[
        "target_team_loss_by_4_plus"
    ].sum()


def test_mirror_audit_passes():
    _, _, _, teams = build_aligned_targets()

    audit = validate_target_symmetry(
        teams
    )

    assert audit[
        "passed"
    ].all()


def test_evidence_alignment_is_exact():
    _, evidence, _, teams = build_aligned_targets()

    comparison = evidence.merge(
        teams[
            [
                "game_pk",
                "team",
            ]
        ],
        on=[
            "game_pk",
            "team",
        ],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    assert comparison[
        "_merge"
    ].eq(
        "both"
    ).all()


def test_targets_are_factual_only():
    _, _, _, teams = build_aligned_targets()

    assert teams[
        "strict_factual_target"
    ].all()

    assert not teams[
        "market_line_used"
    ].any()

    assert not teams[
        "pregame_evidence_included"
    ].any()

    assert not teams[
        "prediction_created"
    ].any()

    assert not teams[
        "same_date_pregame_feature_used"
    ].any()

    assert not teams[
        "future_game_used"
    ].any()


def test_saved_output_matches_evidence_rows():
    evidence = pd.read_parquet(
        EVIDENCE_PATH,
        columns=[
            "game_pk",
            "team",
        ],
    )

    saved = pd.read_parquet(
        OUTPUT_PATH
    )

    assert len(
        saved
    ) == len(
        evidence
    )
