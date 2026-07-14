"""
Read-only game reconstruction for the ATLAS Baseball Brain.

This module joins existing canonical ATLAS products into one
standardized historical game object.

It does not:

- create predictions
- update identities
- alter source data
- write season outputs
- use future games
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    BRAIN_ENGINE_VERSION,
    PHASE_1_INVARIANTS,
    assert_canonical_sources_exist,
)


RECONSTRUCTION_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class GameReconstruction:
    """
    Standardized read-only representation of one historical game.
    """

    game_pk: int
    season: int

    game_core: pd.DataFrame
    manifest: pd.DataFrame
    pregame_teams: pd.DataFrame
    lineups: pd.DataFrame
    bullpens: pd.DataFrame
    events: pd.DataFrame
    targets: pd.DataFrame

    validation: dict[str, Any]
    lineage: dict[str, Any]


def _normalize_game_pk(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    if "game_pk" in result.columns:
        result["game_pk"] = pd.to_numeric(
            result["game_pk"],
            errors="coerce",
        ).astype("Int64")

    return result


def _normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    if "game_date" in result.columns:
        result["game_date"] = pd.to_datetime(
            result["game_date"],
            errors="coerce",
        ).dt.normalize()

    return result


def _normalize_source(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return _normalize_dates(
        _normalize_game_pk(dataframe)
    )


def _load_game_rows(
    path: Path,
    game_pk: int,
) -> pd.DataFrame:
    """
    Read one game's rows from a Parquet source.

    Filters are attempted first. A normal read fallback is used
    for compatibility with files lacking suitable statistics.
    """
    try:
        dataframe = pd.read_parquet(
            path,
            filters=[
                (
                    "game_pk",
                    "==",
                    int(game_pk),
                )
            ],
        )
    except Exception:
        dataframe = pd.read_parquet(path)

        dataframe = _normalize_game_pk(
            dataframe
        )

        dataframe = dataframe[
            dataframe["game_pk"].eq(
                int(game_pk)
            )
        ].copy()

    dataframe = _normalize_source(
        dataframe
    )

    if "game_pk" in dataframe.columns:
        dataframe = dataframe[
            dataframe["game_pk"].eq(
                int(game_pk)
            )
        ].copy()

    return dataframe.reset_index(
        drop=True
    )


def _season_matches(
    dataframe: pd.DataFrame,
    season: int,
) -> bool:
    if dataframe.empty:
        return False

    if "atlas_season" in dataframe.columns:
        values = (
            pd.to_numeric(
                dataframe["atlas_season"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )

        return bool(
            len(values) > 0
            and set(values) == {int(season)}
        )

    if "game_year" in dataframe.columns:
        values = (
            pd.to_numeric(
                dataframe["game_year"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )

        return bool(
            len(values) > 0
            and set(values) == {int(season)}
        )

    if "game_date" in dataframe.columns:
        years = (
            dataframe["game_date"]
            .dropna()
            .dt.year
            .unique()
        )

        return bool(
            len(years) > 0
            and set(years) == {int(season)}
        )

    return True


def _extract_team_set(
    dataframe: pd.DataFrame,
) -> set[str]:
    if (
        dataframe.empty
        or "team" not in dataframe.columns
    ):
        return set()

    return set(
        dataframe["team"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )


def _extract_game_teams(
    game_core: pd.DataFrame,
) -> set[str]:
    if game_core.empty:
        return set()

    teams: set[str] = set()

    for column in [
        "home_team",
        "away_team",
    ]:
        if column not in game_core.columns:
            continue

        values = (
            game_core[column]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
        )

        teams.update(values)

    return teams


def _last_numeric_value(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> int | None:
    for column in columns:
        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if not values.empty:
            return int(values.iloc[-1])

    return None


def _first_numeric_value(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> int | None:
    for column in columns:
        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if not values.empty:
            return int(values.iloc[0])

    return None


def _score_sources(
    game_core: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, dict[str, int | None]]:
    return {
        "game_core": {
            "home_score": _first_numeric_value(
                game_core,
                ["home_score"],
            ),
            "away_score": _first_numeric_value(
                game_core,
                ["away_score"],
            ),
        },
        "events": {
            "home_score": _last_numeric_value(
                events,
                [
                    "post_home_score",
                    "home_score",
                ],
            ),
            "away_score": _last_numeric_value(
                events,
                [
                    "post_away_score",
                    "away_score",
                ],
            ),
        },
        "targets": {
            "home_score": _first_numeric_value(
                targets,
                ["home_score"],
            ),
            "away_score": _first_numeric_value(
                targets,
                ["away_score"],
            ),
        },
    }


def _scores_agree(
    scores: dict[str, dict[str, int | None]],
) -> bool:
    pairs = []

    for source in scores.values():
        home_score = source["home_score"]
        away_score = source["away_score"]

        if (
            home_score is None
            or away_score is None
        ):
            continue

        pairs.append(
            (
                int(home_score),
                int(away_score),
            )
        )

    return bool(
        len(pairs) >= 2
        and len(set(pairs)) == 1
    )


def _boolean_column_is_false(
    dataframe: pd.DataFrame,
    column: str,
) -> bool | None:
    if column not in dataframe.columns:
        return None

    values = (
        dataframe[column]
        .dropna()
    )

    if values.empty:
        return None

    return bool(
        (~values.astype(bool)).all()
    )


def _boolean_column_is_true(
    dataframe: pd.DataFrame,
    column: str,
) -> bool | None:
    if column not in dataframe.columns:
        return None

    values = (
        dataframe[column]
        .dropna()
    )

    if values.empty:
        return None

    return bool(
        values.astype(bool).all()
    )


def _pregame_safety_checks(
    pregame_teams: pd.DataFrame,
    bullpens: pd.DataFrame,
) -> dict[str, bool | None]:
    checks: dict[str, bool | None] = {}

    false_columns = [
        "uses_outcome_statistics",
        "uses_final_score",
        "uses_future_games",
        "current_game_outcome_used",
        "future_games_used",
        "bullpen_current_game_outcome_used",
        "bullpen_future_games_used",
    ]

    true_columns = [
        "strict_backtest_safe",
        "strict_pregame_safe",
        "bullpen_features_strict_pregame_safe",
    ]

    for column in false_columns:
        if column in pregame_teams.columns:
            checks[
                f"pregame.{column}"
            ] = _boolean_column_is_false(
                pregame_teams,
                column,
            )

        if column in bullpens.columns:
            checks[
                f"bullpen.{column}"
            ] = _boolean_column_is_false(
                bullpens,
                column,
            )

    for column in true_columns:
        if column in pregame_teams.columns:
            checks[
                f"pregame.{column}"
            ] = _boolean_column_is_true(
                pregame_teams,
                column,
            )

        if column in bullpens.columns:
            checks[
                f"bullpen.{column}"
            ] = _boolean_column_is_true(
                bullpens,
                column,
            )

    return checks


def _all_available_safety_checks_pass(
    checks: dict[str, bool | None],
) -> bool:
    available = [
        value
        for value in checks.values()
        if value is not None
    ]

    return bool(
        available
        and all(available)
    )


def _sort_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    sort_columns = [
        column
        for column in [
            "at_bat_number",
            "pitch_number",
            "event_index",
            "event_number",
        ]
        if column in events.columns
    ]

    if not sort_columns:
        return events.reset_index(
            drop=True
        )

    return events.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)


def validate_reconstruction(
    *,
    game_pk: int,
    season: int,
    game_core: pd.DataFrame,
    manifest: pd.DataFrame,
    pregame_teams: pd.DataFrame,
    lineups: pd.DataFrame,
    bullpens: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate one reconstructed game without modifying any data.
    """
    expected_team_rows = int(
        PHASE_1_INVARIANTS[
            "normal_game_team_rows"
        ]
    )

    expected_lineup_rows = int(
        PHASE_1_INVARIANTS[
            "normal_game_lineup_rows"
        ]
    )

    expected_bullpen_rows = int(
        PHASE_1_INVARIANTS[
            "normal_game_bullpen_rows"
        ]
    )

    game_teams = _extract_game_teams(
        game_core
    )

    pregame_teams_set = _extract_team_set(
        pregame_teams
    )

    lineup_teams_set = _extract_team_set(
        lineups
    )

    bullpen_teams_set = _extract_team_set(
        bullpens
    )

    scores = _score_sources(
        game_core=game_core,
        events=events,
        targets=targets,
    )

    safety_checks = (
        _pregame_safety_checks(
            pregame_teams=pregame_teams,
            bullpens=bullpens,
        )
    )

    known_exception = int(game_pk) in set(
        PHASE_1_INVARIANTS[
            "known_exception_game_pks"
        ]
    )

    checks: dict[str, Any] = {
        "known_exception":
            known_exception,

        "game_core_rows":
            int(len(game_core)),

        "manifest_rows":
            int(len(manifest)),

        "pregame_team_rows":
            int(len(pregame_teams)),

        "lineup_rows":
            int(len(lineups)),

        "bullpen_rows":
            int(len(bullpens)),

        "event_rows":
            int(len(events)),

        "target_rows":
            int(len(targets)),

        "one_game_core_row":
            len(game_core) == 1,

        "one_manifest_row":
            len(manifest) == 1,

        "two_pregame_team_rows":
            len(pregame_teams)
            == expected_team_rows,

        "two_lineup_rows":
            len(lineups)
            == expected_lineup_rows,

        "two_bullpen_rows":
            len(bullpens)
            == expected_bullpen_rows,

        "events_available":
            len(events) > 0,

        "one_target_row":
            len(targets) == 1,

        "game_core_season_matches":
            _season_matches(
                game_core,
                season,
            ),

        "pregame_season_matches":
            _season_matches(
                pregame_teams,
                season,
            ),

        "lineup_season_matches":
            _season_matches(
                lineups,
                season,
            ),

        "bullpen_season_matches":
            _season_matches(
                bullpens,
                season,
            ),

        "event_season_matches":
            _season_matches(
                events,
                season,
            ),

        "target_season_matches":
            _season_matches(
                targets,
                season,
            ),

        "pregame_teams_match_core":
            bool(
                game_teams
                and pregame_teams_set
                == game_teams
            ),

        "lineup_teams_match_core":
            bool(
                game_teams
                and lineup_teams_set
                == game_teams
            ),

        "bullpen_teams_match_core":
            bool(
                game_teams
                and bullpen_teams_set
                == game_teams
            ),

        "duplicate_pregame_team_rows":
            int(
                pregame_teams.duplicated(
                    subset=[
                        "game_pk",
                        "team",
                    ]
                ).sum()
            )
            if all(
                column in pregame_teams.columns
                for column in [
                    "game_pk",
                    "team",
                ]
            )
            else None,

        "duplicate_lineup_team_rows":
            int(
                lineups.duplicated(
                    subset=[
                        "game_pk",
                        "team",
                    ]
                ).sum()
            )
            if all(
                column in lineups.columns
                for column in [
                    "game_pk",
                    "team",
                ]
            )
            else None,

        "duplicate_bullpen_team_rows":
            int(
                bullpens.duplicated(
                    subset=[
                        "game_pk",
                        "team",
                    ]
                ).sum()
            )
            if all(
                column in bullpens.columns
                for column in [
                    "game_pk",
                    "team",
                ]
            )
            else None,

        "scores":
            scores,

        "scores_agree":
            _scores_agree(scores),

        "pregame_safety_checks":
            safety_checks,

        "all_available_pregame_safety_checks_pass":
            _all_available_safety_checks_pass(
                safety_checks
            ),
    }

    required_boolean_checks = [
        "one_game_core_row",
        "one_manifest_row",
        "two_pregame_team_rows",
        "two_lineup_rows",
        "two_bullpen_rows",
        "events_available",
        "one_target_row",
        "game_core_season_matches",
        "pregame_season_matches",
        "lineup_season_matches",
        "bullpen_season_matches",
        "event_season_matches",
        "target_season_matches",
        "pregame_teams_match_core",
        "lineup_teams_match_core",
        "bullpen_teams_match_core",
        "scores_agree",
        "all_available_pregame_safety_checks_pass",
    ]

    duplicate_checks_pass = all(
        checks[name] == 0
        for name in [
            "duplicate_pregame_team_rows",
            "duplicate_lineup_team_rows",
            "duplicate_bullpen_team_rows",
        ]
    )

    checks[
        "duplicate_checks_pass"
    ] = duplicate_checks_pass

    checks[
        "reconstruction_pass"
    ] = bool(
        all(
            checks[name]
            for name
            in required_boolean_checks
        )
        and duplicate_checks_pass
    )

    return checks


def reconstruct_game(
    game_pk: int,
    season: int,
) -> GameReconstruction:
    """
    Reconstruct one historical game from canonical ATLAS data.
    """
    game_pk = int(game_pk)
    season = int(season)

    paths = assert_canonical_sources_exist(
        season=season,
        require_pitcher_sources=False,
    )

    game_core = _load_game_rows(
        paths["game_card_core"],
        game_pk,
    )

    manifest = _load_game_rows(
        paths["game_card_manifest"],
        game_pk,
    )

    pregame_teams = _load_game_rows(
        paths["pregame_interactions"],
        game_pk,
    )

    lineups = _load_game_rows(
        paths["historical_lineups"],
        game_pk,
    )

    bullpens = _load_game_rows(
        paths["bullpen_pregame_state"],
        game_pk,
    )

    events = _sort_events(
        _load_game_rows(
            paths["game_events"],
            game_pk,
        )
    )

    targets = _load_game_rows(
        paths["game_targets"],
        game_pk,
    )

    validation = validate_reconstruction(
        game_pk=game_pk,
        season=season,
        game_core=game_core,
        manifest=manifest,
        pregame_teams=pregame_teams,
        lineups=lineups,
        bullpens=bullpens,
        events=events,
        targets=targets,
    )

    lineage = {
        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "reconstruction_engine_version":
            RECONSTRUCTION_ENGINE_VERSION,

        "season":
            season,

        "game_pk":
            game_pk,

        "source_paths": {
            name: str(path)
            for name, path in paths.items()
        },

        "read_only":
            True,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "current_game_outcome_used_in_pregame":
            False,

        "future_games_used":
            False,
    }

    return GameReconstruction(
        game_pk=game_pk,
        season=season,
        game_core=game_core,
        manifest=manifest,
        pregame_teams=pregame_teams,
        lineups=lineups,
        bullpens=bullpens,
        events=events,
        targets=targets,
        validation=validation,
        lineage=lineage,
    )
