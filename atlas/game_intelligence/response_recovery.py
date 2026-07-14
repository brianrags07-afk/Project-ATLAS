"""
Factual response-and-recovery summaries for Project ATLAS.

Consumes frozen scoring-event roles and team game-flow facts.

This module describes how teams responded after opponent scoring.
It creates no predictions, identities, or explanations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION


RESPONSE_RECOVERY_VERSION = "1.0.0"

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

TEAM_FLOW_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "team_game_flow"
    / "{season}"
    / "team_game_flow.parquet"
)

OUTPUT_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "response_recovery"
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


def _team_margin_before(
    row: pd.Series,
    team: str,
) -> int:
    if team == row["home_team"]:
        return int(
            row["pre_home_score"]
            - row["pre_away_score"]
        )

    return int(
        row["pre_away_score"]
        - row["pre_home_score"]
    )


def _team_margin_after(
    row: pd.Series,
    team: str,
) -> int:
    if team == row["home_team"]:
        return int(
            row["post_home_score"]
            - row["post_away_score"]
        )

    return int(
        row["post_away_score"]
        - row["post_home_score"]
    )


def _longest_opponent_scoring_streak(
    game: pd.DataFrame,
    team: str,
) -> tuple[int, int]:
    longest_events = 0
    longest_runs = 0

    current_events = 0
    current_runs = 0

    for row in game.itertuples(
        index=False
    ):
        scoring_team = str(
            row.scoring_team
        )

        runs = int(
            row.runs_on_play
        )

        if scoring_team != team:
            current_events += 1
            current_runs += runs

            longest_events = max(
                longest_events,
                current_events,
            )

            longest_runs = max(
                longest_runs,
                current_runs,
            )

        else:
            current_events = 0
            current_runs = 0

    return (
        int(longest_events),
        int(longest_runs),
    )


def _build_team_response_row(
    game: pd.DataFrame,
    team_flow_row: pd.Series,
) -> dict[str, Any]:
    game = game.sort_values(
        "scoring_event_number",
        kind="stable",
    ).reset_index(drop=True)

    team = str(
        team_flow_row["team"]
    )

    opponent = str(
        team_flow_row["opponent"]
    )

    opponent_events = game[
        game["scoring_team"].eq(
            opponent
        )
    ]

    response_records = []

    for opponent_index, opponent_row in opponent_events.iterrows():
        later = game[
            game[
                "scoring_event_number"
            ].gt(
                int(
                    opponent_row[
                        "scoring_event_number"
                    ]
                )
            )
        ]

        if later.empty:
            response_records.append({
                "opponent_event_number":
                    int(
                        opponent_row[
                            "scoring_event_number"
                        ]
                    ),

                "opponent_inning":
                    int(
                        opponent_row[
                            "inning"
                        ]
                    ),

                "team_scored_next":
                    False,

                "team_responded_eventually":
                    False,

                "response_event_number":
                    None,

                "response_inning":
                    None,

                "response_event_gap":
                    None,

                "response_inning_gap":
                    None,

                "same_inning_response":
                    False,

                "within_one_inning_response":
                    False,

                "response_tied_game":
                    False,

                "response_took_lead":
                    False,

                "late_response":
                    False,

                "opponent_events_before_response":
                    int(
                        len(later)
                    ),

                "opponent_runs_before_response":
                    int(
                        later[
                            "runs_on_play"
                        ].sum()
                    ),
            })

            continue

        next_event = later.iloc[0]

        team_scored_next = bool(
            next_event[
                "scoring_team"
            ] == team
        )

        team_later = later[
            later[
                "scoring_team"
            ].eq(team)
        ]

        team_responded_eventually = bool(
            not team_later.empty
        )

        if not team_responded_eventually:
            opponent_only = later[
                later[
                    "scoring_team"
                ].eq(opponent)
            ]

            response_records.append({
                "opponent_event_number":
                    int(
                        opponent_row[
                            "scoring_event_number"
                        ]
                    ),

                "opponent_inning":
                    int(
                        opponent_row[
                            "inning"
                        ]
                    ),

                "team_scored_next":
                    team_scored_next,

                "team_responded_eventually":
                    False,

                "response_event_number":
                    None,

                "response_inning":
                    None,

                "response_event_gap":
                    None,

                "response_inning_gap":
                    None,

                "same_inning_response":
                    False,

                "within_one_inning_response":
                    False,

                "response_tied_game":
                    False,

                "response_took_lead":
                    False,

                "late_response":
                    False,

                "opponent_events_before_response":
                    int(
                        len(opponent_only)
                    ),

                "opponent_runs_before_response":
                    int(
                        opponent_only[
                            "runs_on_play"
                        ].sum()
                    ),
            })

            continue

        response_row = team_later.iloc[0]

        response_number = int(
            response_row[
                "scoring_event_number"
            ]
        )

        opponent_number = int(
            opponent_row[
                "scoring_event_number"
            ]
        )

        response_inning = int(
            response_row[
                "inning"
            ]
        )

        opponent_inning = int(
            opponent_row[
                "inning"
            ]
        )

        between = game[
            game[
                "scoring_event_number"
            ].gt(
                opponent_number
            )
            & game[
                "scoring_event_number"
            ].lt(
                response_number
            )
        ]

        opponent_between = between[
            between[
                "scoring_team"
            ].eq(opponent)
        ]

        post_margin = _team_margin_after(
            response_row,
            team,
        )

        response_records.append({
            "opponent_event_number":
                opponent_number,

            "opponent_inning":
                opponent_inning,

            "team_scored_next":
                team_scored_next,

            "team_responded_eventually":
                True,

            "response_event_number":
                response_number,

            "response_inning":
                response_inning,

            "response_event_gap":
                int(
                    response_number
                    - opponent_number
                ),

            "response_inning_gap":
                int(
                    response_inning
                    - opponent_inning
                ),

            "same_inning_response":
                bool(
                    response_inning
                    == opponent_inning
                ),

            "within_one_inning_response":
                bool(
                    response_inning
                    - opponent_inning
                    <= 1
                ),

            "response_tied_game":
                bool(
                    post_margin == 0
                ),

            "response_took_lead":
                bool(
                    post_margin > 0
                ),

            "late_response":
                bool(
                    response_inning >= 7
                ),

            "opponent_events_before_response":
                int(
                    len(opponent_between)
                ),

            "opponent_runs_before_response":
                int(
                    opponent_between[
                        "runs_on_play"
                    ].sum()
                ),
        })

    response_table = pd.DataFrame(
        response_records
    )

    longest_opponent_events, longest_opponent_runs = (
        _longest_opponent_scoring_streak(
            game,
            team,
        )
    )

    first_scoring_team = str(
        game[
            "scoring_team"
        ].iloc[0]
    )

    allowed_first_score = bool(
        first_scoring_team == opponent
    )

    won = bool(
        team_flow_row[
            "won"
        ]
    )

    lost = bool(
        team_flow_row[
            "lost"
        ]
    )

    if response_table.empty:
        opponent_scoring_events = 0
        team_scored_next_count = 0
        eventual_responses = 0
        same_inning_responses = 0
        within_one_inning_responses = 0
        tying_responses = 0
        go_ahead_responses = 0
        late_responses = 0
        unanswered_opponent_events = 0
        average_response_event_gap = None
        average_response_inning_gap = None

    else:
        opponent_scoring_events = int(
            len(response_table)
        )

        team_scored_next_count = int(
            response_table[
                "team_scored_next"
            ].sum()
        )

        eventual_responses = int(
            response_table[
                "team_responded_eventually"
            ].sum()
        )

        same_inning_responses = int(
            response_table[
                "same_inning_response"
            ].sum()
        )

        within_one_inning_responses = int(
            response_table[
                "within_one_inning_response"
            ].sum()
        )

        tying_responses = int(
            response_table[
                "response_tied_game"
            ].sum()
        )

        go_ahead_responses = int(
            response_table[
                "response_took_lead"
            ].sum()
        )

        late_responses = int(
            response_table[
                "late_response"
            ].sum()
        )

        unanswered_opponent_events = int(
            (
                ~response_table[
                    "team_responded_eventually"
                ]
            ).sum()
        )

        responded = response_table[
            response_table[
                "team_responded_eventually"
            ]
        ]

        average_response_event_gap = (
            float(
                responded[
                    "response_event_gap"
                ].mean()
            )
            if not responded.empty
            else None
        )

        average_response_inning_gap = (
            float(
                responded[
                    "response_inning_gap"
                ].mean()
            )
            if not responded.empty
            else None
        )

    ever_answered_after_opponent_score = bool(
        eventual_responses > 0
    )

    answered_after_falling_behind = False
    tied_after_falling_behind = False
    took_lead_after_falling_behind = False

    for _, row in game.iterrows():
        if row["scoring_team"] != team:
            continue

        pre_margin = _team_margin_before(
            row,
            team,
        )

        post_margin = _team_margin_after(
            row,
            team,
        )

        if pre_margin < 0:
            answered_after_falling_behind = True

            if post_margin == 0:
                tied_after_falling_behind = True

            if post_margin > 0:
                took_lead_after_falling_behind = True

    won_after_allowing_first_score = bool(
        won
        and allowed_first_score
    )

    lost_after_scoring_first = bool(
        lost
        and not allowed_first_score
    )

    immediate_response_rate = (
        float(
            team_scored_next_count
            / opponent_scoring_events
        )
        if opponent_scoring_events > 0
        else None
    )

    eventual_response_rate = (
        float(
            eventual_responses
            / opponent_scoring_events
        )
        if opponent_scoring_events > 0
        else None
    )

    return {
        "game_pk":
            int(
                team_flow_row[
                    "game_pk"
                ]
            ),

        "game_date":
            team_flow_row[
                "game_date"
            ],

        "atlas_season":
            int(
                team_flow_row[
                    "atlas_season"
                ]
            ),

        "team":
            team,

        "opponent":
            opponent,

        "home_away":
            str(
                team_flow_row[
                    "home_away"
                ]
            ),

        "won":
            won,

        "lost":
            lost,

        "team_score":
            int(
                team_flow_row[
                    "team_score"
                ]
            ),

        "opponent_score":
            int(
                team_flow_row[
                    "opponent_score"
                ]
            ),

        "run_differential":
            int(
                team_flow_row[
                    "run_differential"
                ]
            ),

        "scored_first":
            bool(
                team_flow_row[
                    "scored_first"
                ]
            ),

        "allowed_first_score":
            allowed_first_score,

        "won_after_allowing_first_score":
            won_after_allowing_first_score,

        "lost_after_scoring_first":
            lost_after_scoring_first,

        "opponent_scoring_events":
            opponent_scoring_events,

        "team_scored_next_after_opponent_event":
            team_scored_next_count,

        "team_eventually_responded":
            eventual_responses,

        "unanswered_opponent_scoring_events":
            unanswered_opponent_events,

        "same_inning_responses":
            same_inning_responses,

        "within_one_inning_responses":
            within_one_inning_responses,

        "tying_responses":
            tying_responses,

        "go_ahead_responses":
            go_ahead_responses,

        "late_responses":
            late_responses,

        "immediate_response_rate":
            immediate_response_rate,

        "eventual_response_rate":
            eventual_response_rate,

        "average_response_event_gap":
            average_response_event_gap,

        "average_response_inning_gap":
            average_response_inning_gap,

        "longest_opponent_unanswered_event_streak":
            longest_opponent_events,

        "longest_opponent_unanswered_run_streak":
            longest_opponent_runs,

        "ever_answered_after_opponent_score":
            ever_answered_after_opponent_score,

        "answered_after_falling_behind":
            answered_after_falling_behind,

        "tied_after_falling_behind":
            tied_after_falling_behind,

        "took_lead_after_falling_behind":
            took_lead_after_falling_behind,

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

        "response_recovery_version":
            RESPONSE_RECOVERY_VERSION,
    }


def build_team_response_recovery(
    roles: pd.DataFrame,
    team_flow: pd.DataFrame,
    season: int = 2024,
) -> pd.DataFrame:
    roles = roles.copy()
    team_flow = team_flow.copy()

    roles["game_pk"] = pd.to_numeric(
        roles["game_pk"],
        errors="raise",
    ).astype("int64")

    team_flow["game_pk"] = pd.to_numeric(
        team_flow["game_pk"],
        errors="raise",
    ).astype("int64")

    roles["atlas_season"] = pd.to_numeric(
        roles["atlas_season"],
        errors="raise",
    ).astype("int64")

    team_flow["atlas_season"] = pd.to_numeric(
        team_flow["atlas_season"],
        errors="raise",
    ).astype("int64")

    roles = roles[
        roles[
            "atlas_season"
        ].eq(
            int(season)
        )
    ].copy()

    team_flow = team_flow[
        team_flow[
            "atlas_season"
        ].eq(
            int(season)
        )
    ].copy()

    if roles.duplicated(
        subset=[
            "game_pk",
            "scoring_event_number",
        ]
    ).any():
        raise AssertionError(
            "Duplicate scoring-event rows found."
        )

    if team_flow.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).any():
        raise AssertionError(
            "Duplicate team game-flow rows found."
        )

    game_lookup = {
        int(game_pk): game.copy()
        for game_pk, game
        in roles.groupby(
            "game_pk",
            sort=False,
        )
    }

    rows = []

    for team_row in team_flow.itertuples(
        index=False
    ):
        values = pd.Series(
            team_row._asdict()
        )

        game_pk = int(
            values[
                "game_pk"
            ]
        )

        game = game_lookup.get(
            game_pk
        )

        if game is None:
            raise KeyError(
                f"Missing scoring roles for game {game_pk}."
            )

        rows.append(
            _build_team_response_row(
                game,
                values,
            )
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)


def audit_team_response_recovery(
    response_facts: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows = []

    for game_pk, game in response_facts.groupby(
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

        one_scored_first = bool(
            int(
                game[
                    "scored_first"
                ].sum()
            ) == 1
        )

        one_allowed_first = bool(
            int(
                game[
                    "allowed_first_score"
                ].sum()
            ) == 1
        )

        score_mirror = bool(
            int(
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

        response_counts_valid = bool(
            (
                game[
                    "team_scored_next_after_opponent_event"
                ]
                <= game[
                    "opponent_scoring_events"
                ]
            ).all()
            and (
                game[
                    "team_eventually_responded"
                ]
                <= game[
                    "opponent_scoring_events"
                ]
            ).all()
            and (
                game[
                    "same_inning_responses"
                ]
                <= game[
                    "team_eventually_responded"
                ]
            ).all()
            and (
                game[
                    "within_one_inning_responses"
                ]
                <= game[
                    "team_eventually_responded"
                ]
            ).all()
        )

        rate_bounds_valid = bool(
            game[
                "immediate_response_rate"
            ].dropna().between(
                0,
                1,
            ).all()
            and game[
                "eventual_response_rate"
            ].dropna().between(
                0,
                1,
            ).all()
        )

        won_after_first_consistent = bool(
            game[
                "won_after_allowing_first_score"
            ].eq(
                game[
                    "won"
                ]
                & game[
                    "allowed_first_score"
                ]
            ).all()
        )

        lost_after_first_consistent = bool(
            game[
                "lost_after_scoring_first"
            ].eq(
                game[
                    "lost"
                ]
                & game[
                    "scored_first"
                ]
            ).all()
        )

        nonnegative_streaks = bool(
            game[
                "longest_opponent_unanswered_event_streak"
            ].ge(0).all()
            and game[
                "longest_opponent_unanswered_run_streak"
            ].ge(0).all()
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
            and one_scored_first
            and one_allowed_first
            and score_mirror
            and response_counts_valid
            and rate_bounds_valid
            and won_after_first_consistent
            and lost_after_first_consistent
            and nonnegative_streaks
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

            "one_scored_first":
                one_scored_first,

            "one_allowed_first":
                one_allowed_first,

            "score_mirror":
                score_mirror,

            "response_counts_valid":
                response_counts_valid,

            "rate_bounds_valid":
                rate_bounds_valid,

            "won_after_first_consistent":
                won_after_first_consistent,

            "lost_after_first_consistent":
                lost_after_first_consistent,

            "nonnegative_streaks":
                nonnegative_streaks,

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


def run_response_recovery_build(
    season: int = 2024,
) -> dict[str, Any]:
    season = int(season)

    role_path = Path(
        str(
            ROLE_TEMPLATE
        ).format(
            season=season
        )
    )

    team_flow_path = Path(
        str(
            TEAM_FLOW_TEMPLATE
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

    fact_path = (
        output_dir
        / "team_response_recovery.parquet"
    )

    audit_path = (
        output_dir
        / "team_response_recovery_audit.parquet"
    )

    failure_path = (
        output_dir
        / "team_response_recovery_failures.parquet"
    )

    metadata_path = (
        output_dir
        / "team_response_recovery_metadata.json"
    )

    failures = pd.DataFrame(
        columns=[
            "season",
            "error_type",
            "error_message",
        ]
    )

    try:
        roles = pd.read_parquet(
            role_path
        )

        team_flow = pd.read_parquet(
            team_flow_path
        )

        response_facts = build_team_response_recovery(
            roles=roles,
            team_flow=team_flow,
            season=season,
        )

        audit = audit_team_response_recovery(
            response_facts
        )

    except Exception as exc:
        failures = pd.DataFrame([
            {
                "season":
                    season,

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            }
        ])

        response_facts = pd.DataFrame()
        audit = pd.DataFrame()

    duplicate_team_games = int(
        response_facts.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
        if not response_facts.empty
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
        response_facts[
            "game_pk"
        ].nunique()
        if not response_facts.empty
        else 0
    )

    teams = int(
        response_facts[
            "team"
        ].nunique()
        if not response_facts.empty
        else 0
    )

    phase_pass = bool(
        len(response_facts) == 4_856
        and games == 2_428
        and teams == 30
        and failures.empty
        and audit_failures.empty
        and duplicate_team_games == 0
    )

    metadata = {
        "engine":
            "ATLAS Response and Recovery Builder",

        "season":
            season,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "response_recovery_version":
            RESPONSE_RECOVERY_VERSION,

        "team_game_rows":
            int(
                len(response_facts)
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

        "won_after_allowing_first_score":
            int(
                response_facts[
                    "won_after_allowing_first_score"
                ].sum()
                if not response_facts.empty
                else 0
            ),

        "lost_after_scoring_first":
            int(
                response_facts[
                    "lost_after_scoring_first"
                ].sum()
                if not response_facts.empty
                else 0
            ),

        "same_inning_responses":
            int(
                response_facts[
                    "same_inning_responses"
                ].sum()
                if not response_facts.empty
                else 0
            ),

        "tying_responses":
            int(
                response_facts[
                    "tying_responses"
                ].sum()
                if not response_facts.empty
                else 0
            ),

        "go_ahead_responses":
            int(
                response_facts[
                    "go_ahead_responses"
                ].sum()
                if not response_facts.empty
                else 0
            ),

        "phase_2d5_pass":
            phase_pass,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "future_games_used":
            False,

        "built_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": {
            "response_facts":
                str(fact_path),

            "audit":
                str(audit_path),

            "failures":
                str(failure_path),

            "metadata":
                str(metadata_path),
        },
    }

    _atomic_parquet_write(
        response_facts,
        fact_path,
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
        "response_facts":
            response_facts,

        "audit":
            audit,

        "failures":
            failures,

        "metadata":
            metadata,
    }
