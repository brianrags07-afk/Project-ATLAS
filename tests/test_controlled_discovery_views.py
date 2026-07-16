from pathlib import Path

import pandas as pd

from atlas.learning.controlled_discovery_views import (
    build_game_discovery_view,
    build_team_game_discovery_view,
    normalize_inputs,
    select_governed_feature_columns,
)


EVIDENCE_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence.parquet'
)

REGISTRY_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence_column_registry.csv'
)

TARGET_PATH = Path(
    '/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_team_game_learning_targets.parquet'
)


def load_inputs():
    evidence = pd.read_parquet(
        EVIDENCE_PATH
    )

    registry = pd.read_csv(
        REGISTRY_PATH
    )

    targets = pd.read_parquet(
        TARGET_PATH
    )

    evidence, targets = normalize_inputs(
        evidence,
        targets,
    )

    features = select_governed_feature_columns(
        evidence,
        registry,
    )

    return (
        evidence,
        targets,
        features,
    )


def test_governed_features_exist():
    evidence, _, features = load_inputs()

    assert len(
        features
    ) > 0

    assert set(
        features
    ).issubset(
        evidence.columns
    )


def test_team_win_by_2_view_grain():
    evidence, targets, features = load_inputs()

    view = build_team_game_discovery_view(
        evidence,
        targets,
        features,
        "target_team_win_by_2_plus",
    )

    assert len(
        view
    ) == 4_856

    assert view[
        "game_pk"
    ].nunique() == 2_428

    assert view.duplicated(
        [
            "game_pk",
            "team",
        ]
    ).sum() == 0


def test_team_view_has_only_one_label():
    evidence, targets, features = load_inputs()

    view = build_team_game_discovery_view(
        evidence,
        targets,
        features,
        "target_team_win",
    )

    assert "target_label" in view.columns

    assert not any(
        column.startswith(
            "target_"
        )
        for column in view.columns
        if column not in ('target_name', 'target_label')
    )


def test_total_view_counts_games_once():
    evidence, targets, features = load_inputs()

    view = build_game_discovery_view(
        evidence,
        targets,
        features,
        "target_game_total_over_10",
    )

    assert len(
        view
    ) == 2_428

    assert view[
        "game_pk"
    ].nunique() == 2_428

    assert view.duplicated(
        [
            "game_pk",
        ]
    ).sum() == 0

    assert view[
        "shared_game_target_counted_once"
    ].all()


def test_game_view_pairs_home_and_away_features():
    evidence, targets, features = load_inputs()

    view = build_game_discovery_view(
        evidence,
        targets,
        features,
        "target_game_total_7_or_less",
    )

    assert any(
        column.startswith(
            "home__"
        )
        for column in view.columns
    )

    assert any(
        column.startswith(
            "away__"
        )
        for column in view.columns
    )


def test_views_do_not_create_decisions():
    evidence, targets, features = load_inputs()

    team_view = build_team_game_discovery_view(
        evidence,
        targets,
        features,
        "target_team_win_by_2_plus",
    )

    game_view = build_game_discovery_view(
        evidence,
        targets,
        features,
        "target_game_total_over_10",
    )

    for view in [
        team_view,
        game_view,
    ]:
        assert not view[
            "prediction_created"
        ].any()

        assert not view[
            "weight_assigned"
        ].any()

        assert not view[
            "canonical_evidence_modified"
        ].any()
