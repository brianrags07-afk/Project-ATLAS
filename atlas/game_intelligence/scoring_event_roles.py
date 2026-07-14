"""
Factual scoring-event role classification for Project ATLAS.

This module classifies each frozen Phase 2C scoring transition
according to what it did to the score state.

It does not:

- estimate momentum
- assign psychological meaning
- update identities
- create predictions
- use sportsbook information
- use future games
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION


SCORING_EVENT_ROLE_VERSION = "1.0.0"

REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT = REPO_ROOT / "data"

TIMELINE_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "scoring_timelines"
    / "{season}"
    / "scoring_state_timelines.parquet"
)

OUTCOME_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "outcomes"
    / "{season}"
    / "game_outcomes.parquet"
)


def _team_score_state(
    *,
    scoring_side: str,
    pre_home_score: int,
    pre_away_score: int,
    post_home_score: int,
    post_away_score: int,
) -> dict[str, int]:
    """
    Convert home/away score fields into scoring-team perspective.
    """
    scoring_side = str(
        scoring_side
    )

    if scoring_side == "HOME":
        return {
            "pre_team_score":
                int(pre_home_score),

            "pre_opponent_score":
                int(pre_away_score),

            "post_team_score":
                int(post_home_score),

            "post_opponent_score":
                int(post_away_score),
        }

    if scoring_side == "AWAY":
        return {
            "pre_team_score":
                int(pre_away_score),

            "pre_opponent_score":
                int(pre_home_score),

            "post_team_score":
                int(post_away_score),

            "post_opponent_score":
                int(post_home_score),
        }

    raise ValueError(
        f"Unsupported scoring side: {scoring_side}"
    )


def classify_scoring_event_role(
    row: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    """
    Classify one scoring transition from the scoring team's view.

    Primary roles are mutually exclusive:

    - opening_score
    - tying_score
    - go_ahead_score
    - lead_extension
    - deficit_reduction
    """
    values = (
        row
        if isinstance(
            row,
            dict,
        )
        else row.to_dict()
    )

    scoring_side = str(
        values["scoring_side"]
    )

    state = _team_score_state(
        scoring_side=scoring_side,
        pre_home_score=int(
            values["pre_home_score"]
        ),
        pre_away_score=int(
            values["pre_away_score"]
        ),
        post_home_score=int(
            values["post_home_score"]
        ),
        post_away_score=int(
            values["post_away_score"]
        ),
    )

    pre_team_score = state[
        "pre_team_score"
    ]

    pre_opponent_score = state[
        "pre_opponent_score"
    ]

    post_team_score = state[
        "post_team_score"
    ]

    post_opponent_score = state[
        "post_opponent_score"
    ]

    pre_margin = int(
        pre_team_score
        - pre_opponent_score
    )

    post_margin = int(
        post_team_score
        - post_opponent_score
    )

    game_was_scoreless = bool(
        pre_team_score == 0
        and pre_opponent_score == 0
    )

    tied_before = bool(
        pre_margin == 0
    )

    trailing_before = bool(
        pre_margin < 0
    )

    leading_before = bool(
        pre_margin > 0
    )

    tied_after = bool(
        post_margin == 0
    )

    leading_after = bool(
        post_margin > 0
    )

    trailing_after = bool(
        post_margin < 0
    )

    opening_score = bool(
        game_was_scoreless
        and leading_after
    )

    tying_score = bool(
        trailing_before
        and tied_after
    )

    go_ahead_score = bool(
        (
            tied_before
            or trailing_before
        )
        and leading_after
        and not opening_score
    )

    lead_extension = bool(
        leading_before
        and leading_after
        and post_margin > pre_margin
    )

    deficit_reduction = bool(
        trailing_before
        and trailing_after
        and post_margin > pre_margin
    )

    primary_roles = {
        "opening_score":
            opening_score,

        "tying_score":
            tying_score,

        "go_ahead_score":
            go_ahead_score,

        "lead_extension":
            lead_extension,

        "deficit_reduction":
            deficit_reduction,
    }

    active_roles = [
        role
        for role, active
        in primary_roles.items()
        if active
    ]

    if len(active_roles) != 1:
        raise AssertionError(
            "Scoring event must have exactly one primary role. "
            f"Active roles: {active_roles}"
        )

    primary_role = active_roles[0]

    late_inning = bool(
        int(
            values["inning"]
        ) >= 7
    )

    final_inning = bool(
        values.get(
            "terminal_scoring_event",
            False,
        )
    )

    created_two_run_lead = bool(
        post_margin >= 2
        and pre_margin < 2
    )

    created_three_run_lead = bool(
        post_margin >= 3
        and pre_margin < 3
    )

    created_one_run_game = bool(
        abs(post_margin) == 1
        and abs(pre_margin) != 1
    )

    erased_deficit = bool(
        trailing_before
        and not trailing_after
    )

    took_lead_from_deficit = bool(
        trailing_before
        and leading_after
    )

    return {
        "pre_scoring_team_score":
            pre_team_score,

        "pre_opponent_score":
            pre_opponent_score,

        "post_scoring_team_score":
            post_team_score,

        "post_opponent_score":
            post_opponent_score,

        "pre_scoring_team_margin":
            pre_margin,

        "post_scoring_team_margin":
            post_margin,

        "primary_scoring_role":
            primary_role,

        "opening_score":
            opening_score,

        "tying_score":
            tying_score,

        "go_ahead_score":
            go_ahead_score,

        "lead_extension":
            lead_extension,

        "deficit_reduction":
            deficit_reduction,

        "erased_deficit":
            erased_deficit,

        "took_lead_from_deficit":
            took_lead_from_deficit,

        "created_one_run_game":
            created_one_run_game,

        "created_two_run_lead":
            created_two_run_lead,

        "created_three_run_lead":
            created_three_run_lead,

        "late_inning_scoring_event":
            late_inning,

        "terminal_scoring_event_role":
            final_inning,

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

        "scoring_event_role_version":
            SCORING_EVENT_ROLE_VERSION,
    }


def classify_scoring_timeline(
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach factual scoring roles to one or more scoring timelines.
    """
    required = {
        "game_pk",
        "scoring_event_number",
        "scoring_side",
        "scoring_team",
        "inning",
        "pre_home_score",
        "pre_away_score",
        "post_home_score",
        "post_away_score",
        "terminal_scoring_event",
    }

    missing = required - set(
        timeline.columns
    )

    if missing:
        raise KeyError(
            "Scoring timeline missing required columns: "
            f"{sorted(missing)}"
        )

    classified_rows = []

    for row in timeline.itertuples(
        index=False
    ):
        values = row._asdict()

        role = classify_scoring_event_role(
            values
        )

        classified_rows.append({
            **values,
            **role,
        })

    classified = pd.DataFrame(
        classified_rows
    )

    if classified.empty:
        return classified

    role_columns = [
        "opening_score",
        "tying_score",
        "go_ahead_score",
        "lead_extension",
        "deficit_reduction",
    ]

    if not classified[
        role_columns
    ].sum(
        axis=1
    ).eq(1).all():
        raise AssertionError(
            "Every scoring transition must have exactly "
            "one primary role."
        )

    if not classified[
        "batting_side"
    ].eq(
        classified[
            "scoring_side"
        ]
    ).all():
        raise AssertionError(
            "Canonical batting side and scoring side differ."
        )

    return classified.sort_values(
        [
            "game_date",
            "game_pk",
            "scoring_event_number",
        ],
        kind="stable",
    ).reset_index(drop=True)


def attach_decisive_scoring_flags(
    classified: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mark the final scoring transition that permanently established
    the winning team's lead.

    This is factual hindsight classification and must never be used
    as a pregame feature.
    """
    if classified.empty:
        return classified.copy()

    required_outcomes = {
        "game_pk",
        "winner_team",
    }

    missing = required_outcomes - set(
        outcomes.columns
    )

    if missing:
        raise KeyError(
            "Outcome table missing required columns: "
            f"{sorted(missing)}"
        )

    outcome_lookup = outcomes[
        [
            "game_pk",
            "winner_team",
        ]
    ].copy()

    result = classified.merge(
        outcome_lookup,
        on="game_pk",
        how="left",
        validate="many_to_one",
    )

    if result[
        "winner_team"
    ].isna().any():
        raise AssertionError(
            "One or more scoring rows are missing a winner."
        )

    result[
        "winning_team_scoring_event"
    ] = result[
        "scoring_team"
    ].eq(
        result[
            "winner_team"
        ]
    )

    result[
        "winner_leading_after_event"
    ] = (
        (
            result[
                "winner_team"
            ].eq(
                result[
                    "home_team"
                ]
            )
            & result[
                "post_home_score"
            ].gt(
                result[
                    "post_away_score"
                ]
            )
        )
        |
        (
            result[
                "winner_team"
            ].eq(
                result[
                    "away_team"
                ]
            )
            & result[
                "post_away_score"
            ].gt(
                result[
                    "post_home_score"
                ]
            )
        )
    )

    result[
        "winner_trailed_or_tied_before_event"
    ] = (
        (
            result[
                "winner_team"
            ].eq(
                result[
                    "home_team"
                ]
            )
            & result[
                "pre_home_score"
            ].le(
                result[
                    "pre_away_score"
                ]
            )
        )
        |
        (
            result[
                "winner_team"
            ].eq(
                result[
                    "away_team"
                ]
            )
            & result[
                "pre_away_score"
            ].le(
                result[
                    "pre_home_score"
                ]
            )
        )
    )

    result[
        "candidate_decisive_score"
    ] = (
        result[
            "winning_team_scoring_event"
        ]
        & result[
            "winner_leading_after_event"
        ]
        & result[
            "winner_trailed_or_tied_before_event"
        ]
    )

    result[
        "decisive_scoring_event"
    ] = False

    for game_pk, game in result.groupby(
        "game_pk",
        sort=False,
    ):
        winner = str(
            game[
                "winner_team"
            ].iloc[0]
        )

        game = game.sort_values(
            "scoring_event_number",
            kind="stable",
        )

        decisive_index = None

        for index, row in game.iterrows():
            if not bool(
                row[
                    "candidate_decisive_score"
                ]
            ):
                continue

            later = game[
                game[
                    "scoring_event_number"
                ].gt(
                    row[
                        "scoring_event_number"
                    ]
                )
            ]

            if winner == row["home_team"]:
                winner_never_lost_lead = bool(
                    (
                        later[
                            "post_home_score"
                        ]
                        > later[
                            "post_away_score"
                        ]
                    ).all()
                )

            else:
                winner_never_lost_lead = bool(
                    (
                        later[
                            "post_away_score"
                        ]
                        > later[
                            "post_home_score"
                        ]
                    ).all()
                )

            if winner_never_lost_lead:
                decisive_index = index
                break

        if decisive_index is None:
            raise AssertionError(
                f"No decisive scoring event found for game {game_pk}."
            )

        result.loc[
            decisive_index,
            "decisive_scoring_event",
        ] = True

    decisive_counts = (
        result.groupby(
            "game_pk",
            sort=False,
        )[
            "decisive_scoring_event"
        ].sum()
    )

    if not decisive_counts.eq(1).all():
        raise AssertionError(
            "Every game must have one decisive scoring event."
        )

    result[
        "postgame_hindsight_only"
    ] = True

    return result
