"""
Team-perspective factual game-flow summaries for Project ATLAS.

Consumes frozen Phase 2D scoring-event roles and creates two rows
per game: one from each team's perspective.

This is postgame factual classification only.

It does not:
- create predictions
- calculate betting value
- update identities
- create explanations
- use future games
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import time

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION


TEAM_GAME_FLOW_VERSION = "1.0.0"

REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT = REPO_ROOT / "data"

ROLE_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "scoring_event_roles"
    / "{season}"
    / "scoring_event_roles.parquet"
)

OUTPUT_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "team_game_flow"
    / "{season}"
)


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        destination,
    )


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=str,
        )

        temporary_name = handle.name

    os.replace(
        temporary_name,
        destination,
    )


def _normalize_roles(
    roles: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "winner_team",
        "scoring_event_number",
        "inning",
        "scoring_team",
        "pre_home_score",
        "pre_away_score",
        "post_home_score",
        "post_away_score",
        "runs_on_play",
        "opening_score",
        "tying_score",
        "go_ahead_score",
        "lead_extension",
        "deficit_reduction",
        "created_two_run_lead",
        "created_three_run_lead",
        "late_inning_scoring_event",
        "decisive_scoring_event",
        "terminal_scoring_event",
        "postgame_hindsight_only",
    }

    missing = required - set(
        roles.columns
    )

    if missing:
        raise KeyError(
            "Scoring-event roles missing required columns: "
            f"{sorted(missing)}"
        )

    normalized = roles.copy()

    integer_columns = [
        "game_pk",
        "atlas_season",
        "scoring_event_number",
        "inning",
        "pre_home_score",
        "pre_away_score",
        "post_home_score",
        "post_away_score",
        "runs_on_play",
    ]

    for column in integer_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="raise",
        ).astype("int64")

    normalized["game_date"] = pd.to_datetime(
        normalized["game_date"],
        errors="raise",
    ).dt.normalize()

    normalized = normalized[
        normalized["atlas_season"].eq(
            int(season)
        )
    ].copy()

    if normalized.duplicated(
        subset=[
            "game_pk",
            "scoring_event_number",
        ]
    ).any():
        raise AssertionError(
            "Duplicate scoring-event role rows found."
        )

    if not normalized[
        "postgame_hindsight_only"
    ].fillna(False).all():
        raise AssertionError(
            "Scoring roles are missing hindsight provenance."
        )

    return normalized.sort_values(
        [
            "game_date",
            "game_pk",
            "scoring_event_number",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _team_margin_series(
    game: pd.DataFrame,
    team: str,
) -> pd.Series:
    if team == game[
        "home_team"
    ].iloc[0]:
        return (
            game[
                "post_home_score"
            ]
            - game[
                "post_away_score"
            ]
        )

    return (
        game[
            "post_away_score"
        ]
        - game[
            "post_home_score"
        ]
    )


def _team_final_score(
    game: pd.DataFrame,
    team: str,
) -> tuple[int, int]:
    final_row = game.sort_values(
        "scoring_event_number",
        kind="stable",
    ).iloc[-1]

    if team == final_row[
        "home_team"
    ]:
        return (
            int(
                final_row[
                    "post_home_score"
                ]
            ),
            int(
                final_row[
                    "post_away_score"
                ]
            ),
        )

    return (
        int(
            final_row[
                "post_away_score"
            ]
        ),
        int(
            final_row[
                "post_home_score"
            ]
        ),
    )


def _winner_post_decisive_facts(
    game: pd.DataFrame,
) -> dict[str, Any]:
    decisive = game[
        game[
            "decisive_scoring_event"
        ]
    ]

    if len(decisive) != 1:
        raise AssertionError(
            "Expected exactly one decisive event."
        )

    decisive_row = decisive.iloc[0]

    winner = str(
        game[
            "winner_team"
        ].iloc[0]
    )

    decisive_number = int(
        decisive_row[
            "scoring_event_number"
        ]
    )

    after_decisive = game[
        game[
            "scoring_event_number"
        ].ge(
            decisive_number
        )
    ].copy()

    winner_margins = _team_margin_series(
        after_decisive,
        winner,
    )

    decisive_margin = int(
        winner_margins.iloc[0]
    )

    minimum_margin_after_decisive = int(
        winner_margins.min()
    )

    returned_to_one_run_margin = bool(
        decisive_margin >= 2
        and winner_margins.eq(1).any()
    )

    opponent_got_within_one = bool(
        winner_margins.le(1).any()
    )

    winner_tied_or_trailed = bool(
        winner_margins.le(0).any()
    )

    winner_additional_scoring = game[
        game[
            "scoring_event_number"
        ].gt(
            decisive_number
        )
        & game[
            "scoring_team"
        ].eq(
            winner
        )
    ]

    winner_additional_runs = int(
        winner_additional_scoring[
            "runs_on_play"
        ].sum()
    )

    winner_additional_events = int(
        len(
            winner_additional_scoring
        )
    )

    return {
        "decisive_scoring_event_number":
            decisive_number,

        "decisive_inning":
            int(
                decisive_row[
                    "inning"
                ]
            ),

        "decisive_scoring_team":
            str(
                decisive_row[
                    "scoring_team"
                ]
            ),

        "decisive_lead_size":
            decisive_margin,

        "winner_minimum_margin_after_decisive":
            minimum_margin_after_decisive,

        "winner_returned_to_one_run_margin":
            returned_to_one_run_margin,

        "opponent_got_within_one_after_decisive":
            opponent_got_within_one,

        "winner_tied_or_trailed_after_decisive":
            winner_tied_or_trailed,

        "winner_additional_scoring_events_after_decisive":
            winner_additional_events,

        "winner_additional_runs_after_decisive":
            winner_additional_runs,
    }


def _build_team_row(
    game: pd.DataFrame,
    team: str,
) -> dict[str, Any]:
    game = game.sort_values(
        "scoring_event_number",
        kind="stable",
    )

    home_team = str(
        game[
            "home_team"
        ].iloc[0]
    )

    away_team = str(
        game[
            "away_team"
        ].iloc[0]
    )

    winner_team = str(
        game[
            "winner_team"
        ].iloc[0]
    )

    if team == home_team:
        opponent = away_team
        home_away = "HOME"

    elif team == away_team:
        opponent = home_team
        home_away = "AWAY"

    else:
        raise ValueError(
            f"Team {team} is not in game."
        )

    team_score, opponent_score = (
        _team_final_score(
            game,
            team,
        )
    )

    run_differential = int(
        team_score
        - opponent_score
    )

    won = bool(
        team == winner_team
    )

    lost = bool(
        not won
    )

    team_events = game[
        game[
            "scoring_team"
        ].eq(team)
    ].copy()

    opponent_events = game[
        game[
            "scoring_team"
        ].eq(opponent)
    ].copy()

    margins = _team_margin_series(
        game,
        team,
    )

    maximum_lead = int(
        max(
            0,
            int(
                margins.max()
            ),
        )
    )

    maximum_deficit = int(
        max(
            0,
            int(
                -margins.min()
            ),
        )
    )

    first_scoring_team = str(
        game[
            "scoring_team"
        ].iloc[0]
    )

    decisive_owner = bool(
        team_events[
            "decisive_scoring_event"
        ].any()
    )

    post_decisive = (
        _winner_post_decisive_facts(
            game
        )
    )

    won_by_2_plus = bool(
        won
        and run_differential >= 2
    )

    won_by_3_plus = bool(
        won
        and run_differential >= 3
    )

    lost_by_2_plus = bool(
        lost
        and run_differential <= -2
    )

    lost_by_3_plus = bool(
        lost
        and run_differential <= -3
    )

    covered_minus_1_5 = bool(
        run_differential >= 2
    )

    covered_plus_1_5 = bool(
        run_differential >= -1
    )

    failed_minus_1_5_as_winner = bool(
        won
        and run_differential == 1
    )

    team_late_events = team_events[
        team_events[
            "late_inning_scoring_event"
        ]
    ]

    opponent_late_events = opponent_events[
        opponent_events[
            "late_inning_scoring_event"
        ]
    ]

    team_late_lead_extensions = team_events[
        team_events[
            "late_inning_scoring_event"
        ]
        & team_events[
            "lead_extension"
        ]
    ]

    team_two_run_creations = team_events[
        team_events[
            "created_two_run_lead"
        ]
    ]

    team_three_run_creations = team_events[
        team_events[
            "created_three_run_lead"
        ]
    ]

    return {
        "game_pk":
            int(
                game[
                    "game_pk"
                ].iloc[0]
            ),

        "game_date":
            game[
                "game_date"
            ].iloc[0],

        "atlas_season":
            int(
                game[
                    "atlas_season"
                ].iloc[0]
            ),

        "team":
            team,

        "opponent":
            opponent,

        "home_away":
            home_away,

        "team_score":
            team_score,

        "opponent_score":
            opponent_score,

        "run_differential":
            run_differential,

        "won":
            won,

        "lost":
            lost,

        "won_by_2_plus":
            won_by_2_plus,

        "won_by_3_plus":
            won_by_3_plus,

        "lost_by_2_plus":
            lost_by_2_plus,

        "lost_by_3_plus":
            lost_by_3_plus,

        "covered_minus_1_5":
            covered_minus_1_5,

        "covered_plus_1_5":
            covered_plus_1_5,

        "failed_minus_1_5_as_winner":
            failed_minus_1_5_as_winner,

        "scored_first":
            first_scoring_team == team,

        "allowed_first_score":
            first_scoring_team == opponent,

        "team_scoring_events":
            int(
                len(team_events)
            ),

        "opponent_scoring_events":
            int(
                len(opponent_events)
            ),

        "team_runs_on_scoring_events":
            int(
                team_events[
                    "runs_on_play"
                ].sum()
            ),

        "opponent_runs_on_scoring_events":
            int(
                opponent_events[
                    "runs_on_play"
                ].sum()
            ),

        "team_opening_scores":
            int(
                team_events[
                    "opening_score"
                ].sum()
            ),

        "team_tying_scores":
            int(
                team_events[
                    "tying_score"
                ].sum()
            ),

        "team_go_ahead_scores":
            int(
                team_events[
                    "go_ahead_score"
                ].sum()
            ),

        "team_lead_extensions":
            int(
                team_events[
                    "lead_extension"
                ].sum()
            ),

        "team_deficit_reductions":
            int(
                team_events[
                    "deficit_reduction"
                ].sum()
            ),

        "opponent_tying_scores":
            int(
                opponent_events[
                    "tying_score"
                ].sum()
            ),

        "opponent_go_ahead_scores":
            int(
                opponent_events[
                    "go_ahead_score"
                ].sum()
            ),

        "opponent_lead_extensions":
            int(
                opponent_events[
                    "lead_extension"
                ].sum()
            ),

        "opponent_deficit_reductions":
            int(
                opponent_events[
                    "deficit_reduction"
                ].sum()
            ),

        "team_late_scoring_events":
            int(
                len(
                    team_late_events
                )
            ),

        "team_late_runs":
            int(
                team_late_events[
                    "runs_on_play"
                ].sum()
            ),

        "opponent_late_scoring_events":
            int(
                len(
                    opponent_late_events
                )
            ),

        "opponent_late_runs":
            int(
                opponent_late_events[
                    "runs_on_play"
                ].sum()
            ),

        "team_late_lead_extensions":
            int(
                len(
                    team_late_lead_extensions
                )
            ),

        "team_created_two_run_lead_events":
            int(
                len(
                    team_two_run_creations
                )
            ),

        "team_created_three_run_lead_events":
            int(
                len(
                    team_three_run_creations
                )
            ),

        "ever_created_two_run_lead":
            bool(
                len(
                    team_two_run_creations
                ) > 0
            ),

        "ever_created_three_run_lead":
            bool(
                len(
                    team_three_run_creations
                ) > 0
            ),

        "maximum_lead":
            maximum_lead,

        "maximum_deficit":
            maximum_deficit,

        "decisive_score_for":
            decisive_owner,

        "decisive_score_against":
            not decisive_owner,

        "decisive_scoring_event_number":
            post_decisive[
                "decisive_scoring_event_number"
            ],

        "decisive_inning":
            post_decisive[
                "decisive_inning"
            ],

        "decisive_scoring_team":
            post_decisive[
                "decisive_scoring_team"
            ],

        "decisive_lead_size":
            post_decisive[
                "decisive_lead_size"
            ],

        "winner_minimum_margin_after_decisive":
            post_decisive[
                "winner_minimum_margin_after_decisive"
            ],

        "winner_returned_to_one_run_margin":
            post_decisive[
                "winner_returned_to_one_run_margin"
            ],

        "opponent_got_within_one_after_decisive":
            post_decisive[
                "opponent_got_within_one_after_decisive"
            ],

        "winner_tied_or_trailed_after_decisive":
            post_decisive[
                "winner_tied_or_trailed_after_decisive"
            ],

        "winner_additional_scoring_events_after_decisive":
            post_decisive[
                "winner_additional_scoring_events_after_decisive"
            ],

        "winner_additional_runs_after_decisive":
            post_decisive[
                "winner_additional_runs_after_decisive"
            ],

        "postgame_factual_only":
            True,

        "pregame_feature_safe":
            False,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "explanation_created":
            False,

        "future_games_used":
            False,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "team_game_flow_version":
            TEAM_GAME_FLOW_VERSION,
    }


def build_team_game_flow(
    roles: pd.DataFrame,
    season: int = 2024,
) -> pd.DataFrame:
    roles = _normalize_roles(
        roles,
        season=season,
    )

    rows = []

    for _, game in roles.groupby(
        "game_pk",
        sort=False,
    ):
        home_team = str(
            game[
                "home_team"
            ].iloc[0]
        )

        away_team = str(
            game[
                "away_team"
            ].iloc[0]
        )

        rows.append(
            _build_team_row(
                game,
                away_team,
            )
        )

        rows.append(
            _build_team_row(
                game,
                home_team,
            )
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)


def audit_team_game_flow(
    team_flow: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows = []

    for game_pk, game in team_flow.groupby(
        "game_pk",
        sort=False,
    ):
        exactly_two_rows = bool(
            len(game) == 2
        )

        one_winner = bool(
            int(
                game[
                    "won"
                ].sum()
            ) == 1
        )

        one_loser = bool(
            int(
                game[
                    "lost"
                ].sum()
            ) == 1
        )

        score_mirror = bool(
            len(game) == 2
            and int(
                game[
                    "team_score"
                ].iloc[0]
            )
            == int(
                game[
                    "opponent_score"
                ].iloc[1]
            )
            and int(
                game[
                    "opponent_score"
                ].iloc[0]
            )
            == int(
                game[
                    "team_score"
                ].iloc[1]
            )
        )

        margin_mirror = bool(
            len(game) == 2
            and int(
                game[
                    "run_differential"
                ].sum()
            ) == 0
        )

        one_scored_first = bool(
            int(
                game[
                    "scored_first"
                ].sum()
            ) == 1
        )

        one_decisive_for = bool(
            int(
                game[
                    "decisive_score_for"
                ].sum()
            ) == 1
        )

        one_decisive_against = bool(
            int(
                game[
                    "decisive_score_against"
                ].sum()
            ) == 1
        )

        run_line_minus_consistent = bool(
            game[
                "covered_minus_1_5"
            ].eq(
                game[
                    "run_differential"
                ].ge(2)
            ).all()
        )

        run_line_plus_consistent = bool(
            game[
                "covered_plus_1_5"
            ].eq(
                game[
                    "run_differential"
                ].ge(-1)
            ).all()
        )

        winner_two_plus_consistent = bool(
            game[
                "won_by_2_plus"
            ].eq(
                game[
                    "won"
                ]
                & game[
                    "run_differential"
                ].ge(2)
            ).all()
        )

        failed_minus_consistent = bool(
            game[
                "failed_minus_1_5_as_winner"
            ].eq(
                game[
                    "won"
                ]
                & game[
                    "run_differential"
                ].eq(1)
            ).all()
        )

        no_winner_reversal = bool(
            (
                ~game[
                    "winner_tied_or_trailed_after_decisive"
                ]
            ).all()
        )

        provenance_pass = bool(
            game[
                "postgame_factual_only"
            ].all()
            and (
                ~game[
                    "pregame_feature_safe"
                ]
            ).all()
            and (
                ~game[
                    "prediction_created"
                ]
            ).all()
            and (
                ~game[
                    "identity_updated"
                ]
            ).all()
            and (
                ~game[
                    "explanation_created"
                ]
            ).all()
            and (
                ~game[
                    "future_games_used"
                ]
            ).all()
        )

        audit_pass = bool(
            exactly_two_rows
            and one_winner
            and one_loser
            and score_mirror
            and margin_mirror
            and one_scored_first
            and one_decisive_for
            and one_decisive_against
            and run_line_minus_consistent
            and run_line_plus_consistent
            and winner_two_plus_consistent
            and failed_minus_consistent
            and no_winner_reversal
            and provenance_pass
        )

        audit_rows.append({
            "game_pk":
                int(game_pk),

            "game_date":
                game[
                    "game_date"
                ].iloc[0],

            "rows":
                int(len(game)),

            "exactly_two_rows":
                exactly_two_rows,

            "one_winner":
                one_winner,

            "one_loser":
                one_loser,

            "score_mirror":
                score_mirror,

            "margin_mirror":
                margin_mirror,

            "one_scored_first":
                one_scored_first,

            "one_decisive_for":
                one_decisive_for,

            "one_decisive_against":
                one_decisive_against,

            "run_line_minus_consistent":
                run_line_minus_consistent,

            "run_line_plus_consistent":
                run_line_plus_consistent,

            "winner_two_plus_consistent":
                winner_two_plus_consistent,

            "failed_minus_consistent":
                failed_minus_consistent,

            "no_winner_reversal":
                no_winner_reversal,

            "provenance_pass":
                provenance_pass,

            "audit_pass":
                audit_pass,
        })

    return pd.DataFrame(
        audit_rows
    ).sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_team_game_flow_build(
    season: int = 2024,
) -> dict[str, Any]:
    started = time.time()
    season = int(season)

    role_path = Path(
        str(
            ROLE_TEMPLATE
        ).format(
            season=season
        )
    )

    output_dir = Path(
        str(
            OUTPUT_TEMPLATE
        ).format(
            season=season
        )
    )

    team_flow_path = (
        output_dir
        / "team_game_flow.parquet"
    )

    audit_path = (
        output_dir
        / "team_game_flow_audit.parquet"
    )

    failure_path = (
        output_dir
        / "team_game_flow_failures.parquet"
    )

    metadata_path = (
        output_dir
        / "team_game_flow_metadata.json"
    )

    failure_records = []

    try:
        roles = pd.read_parquet(
            role_path
        )

        team_flow = build_team_game_flow(
            roles,
            season=season,
        )

        audit = audit_team_game_flow(
            team_flow
        )

    except Exception as exc:
        failure_records.append({
            "season":
                season,

            "error_type":
                type(exc).__name__,

            "error_message":
                str(exc),
        })

        team_flow = pd.DataFrame()
        audit = pd.DataFrame()

    failures = pd.DataFrame(
        failure_records,
        columns=[
            "season",
            "error_type",
            "error_message",
        ],
    )

    duplicate_team_games = int(
        team_flow.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
        if not team_flow.empty
        else 0
    )

    audit_failures = (
        audit[
            ~audit[
                "audit_pass"
            ]
        ]
        if not audit.empty
        else pd.DataFrame()
    )

    games = int(
        team_flow[
            "game_pk"
        ].nunique()
        if not team_flow.empty
        else 0
    )

    teams = int(
        team_flow[
            "team"
        ].nunique()
        if not team_flow.empty
        else 0
    )

    phase_pass = bool(
        len(team_flow) == 4_856
        and games == 2_428
        and teams == 30
        and failures.empty
        and audit_failures.empty
        and duplicate_team_games == 0
    )

    elapsed = (
        time.time()
        - started
    )

    metadata = {
        "engine":
            "ATLAS Team-Perspective Game-Flow Builder",

        "season":
            season,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "team_game_flow_version":
            TEAM_GAME_FLOW_VERSION,

        "team_game_rows":
            int(
                len(team_flow)
            ),

        "games":
            games,

        "teams":
            teams,

        "build_failures":
            int(
                len(failures)
            ),

        "audit_failures":
            int(
                len(audit_failures)
            ),

        "duplicate_team_games":
            duplicate_team_games,

        "wins_by_2_plus":
            int(
                team_flow[
                    "won_by_2_plus"
                ].sum()
                if not team_flow.empty
                else 0
            ),

        "wins_by_3_plus":
            int(
                team_flow[
                    "won_by_3_plus"
                ].sum()
                if not team_flow.empty
                else 0
            ),

        "one_run_winners":
            int(
                team_flow[
                    "failed_minus_1_5_as_winner"
                ].sum()
                if not team_flow.empty
                else 0
            ),

        "minus_1_5_covers":
            int(
                team_flow[
                    "covered_minus_1_5"
                ].sum()
                if not team_flow.empty
                else 0
            ),

        "plus_1_5_covers":
            int(
                team_flow[
                    "covered_plus_1_5"
                ].sum()
                if not team_flow.empty
                else 0
            ),

        "phase_2d3_pass":
            phase_pass,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "future_games_used":
            False,

        "elapsed_seconds":
            float(elapsed),

        "built_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": {
            "team_flow":
                str(team_flow_path),

            "audit":
                str(audit_path),

            "failures":
                str(failure_path),

            "metadata":
                str(metadata_path),
        },
    }

    _atomic_parquet_write(
        team_flow,
        team_flow_path,
    )

    _atomic_parquet_write(
        audit,
        audit_path,
    )

    _atomic_parquet_write(
        failures,
        failure_path,
    )

    _atomic_json_write(
        metadata,
        metadata_path,
    )

    return {
        "team_flow":
            team_flow,

        "audit":
            audit,

        "failures":
            failures,

        "metadata":
            metadata,
    }
