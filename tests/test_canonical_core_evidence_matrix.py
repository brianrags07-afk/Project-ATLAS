from pathlib import Path

import pandas as pd

from atlas.pregame.canonical_core_evidence_matrix import (
    ENGINE_VERSION,
    build_canonical_core_evidence,
)


ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

IDENTITY_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/pregame_identity_matchups/2024/pregame_identity_matchups.parquet'
)

BULLPEN_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/clean_bullpen_pregame_facts/2024/clean_bullpen_pregame_facts.parquet'
)

LINEUP_STARTER_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/clean_starter_lineup_pregame/lineup_starter_inputs/2024/lineup_starter_inputs.parquet'
)

OUTPUT_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence.parquet'
)


def build_matrix():
    identity = pd.read_parquet(
        IDENTITY_PATH
    )

    bullpen = pd.read_parquet(
        BULLPEN_PATH
    )

    lineup_starter = pd.read_parquet(
        LINEUP_STARTER_PATH
    )

    return build_canonical_core_evidence(
        identity=identity,
        bullpen=bullpen,
        lineup_starter=lineup_starter,
        season=2024,
    )


def test_engine_version():
    assert ENGINE_VERSION == "1.0.0"


def test_required_sources_exist():
    assert IDENTITY_PATH.exists()
    assert BULLPEN_PATH.exists()
    assert LINEUP_STARTER_PATH.exists()


def test_matrix_has_expected_grain():
    matrix, _, _, _ = build_matrix()

    assert len(
        matrix
    ) == 4_856

    assert matrix[
        "game_pk"
    ].nunique() == 2_428

    assert matrix[
        "team"
    ].nunique() == 30


def test_matrix_has_no_duplicate_team_games():
    matrix, _, _, _ = build_matrix()

    duplicates = matrix.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).sum()

    assert duplicates == 0


def test_matrix_has_exactly_two_rows_per_game():
    matrix, _, _, _ = build_matrix()

    assert matrix.groupby(
        "game_pk"
    ).size().eq(
        2
    ).all()


def test_all_source_families_are_present():
    matrix, _, _, _ = build_matrix()

    assert any(
        column.startswith(
            "identity__"
        )
        for column in matrix.columns
    )

    assert any(
        column.startswith(
            "bullpen__"
        )
        for column in matrix.columns
    )

    assert any(
        column.startswith(
            "lineup_starter__"
        )
        for column in matrix.columns
    )


def test_join_audit_passes():
    _, _, join_audit, _ = build_matrix()

    assert join_audit[
        "passed"
    ].all()


def test_matrix_is_pregame_safe():
    matrix, _, _, _ = build_matrix()

    assert matrix[
        "strict_backtest_safe"
    ].all()

    assert not matrix[
        "same_date_games_used"
    ].any()

    assert not matrix[
        "future_games_used"
    ].any()


def test_no_decisions_or_targets_created():
    matrix, _, _, _ = build_matrix()

    assert not matrix[
        "handcrafted_scores_included"
    ].any()

    assert not matrix[
        "prediction_values_created"
    ].any()

    assert not matrix[
        "market_used"
    ].any()

    assert not matrix[
        "target_columns_included"
    ].any()


def test_column_registry_covers_output():
    matrix, registry, _, _ = build_matrix()

    assert set(
        matrix.columns
    ).issubset(
        set(
            registry[
                "output_column"
            ]
        )
    )


def test_saved_output_exists():
    assert OUTPUT_PATH.exists()

    saved = pd.read_parquet(
        OUTPUT_PATH
    )

    assert len(
        saved
    ) == 4_856
