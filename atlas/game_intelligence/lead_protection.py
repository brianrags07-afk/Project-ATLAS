"""
Factual lead-protection and run-line separation summaries.

This module consumes frozen scoring-event roles and frozen
team-perspective game-flow facts.

It records what happened after teams obtained leads.

It does not:
- predict future games
- estimate betting value
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

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION


LEAD_PROTECTION_VERSION = "1.0.0"

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
    / "lead_protection"
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


def _team_margin(
    game: pd.DataFrame,
    team: str,
) -> pd.Series:
    home_team = str(
        game["home_team"].iloc[0]
    )

    if team == home_team:
        return (
            game["post_home_score"]
            - game["post_away_score"]
        )

    return (
        game["post_away_score"]
        - game["post_home_score"]
    )


def _first_event_number(
    game: pd.DataFrame,
    mask: pd.Series,
) -> int | None:
    matching = game.loc[
        mask,
        "scoring_event_number",
    ]

    if matching.empty:
        return None

    return int(
        matching.iloc[0]
    )


def _first_inning(
    game: pd.DataFrame,
    mask: pd.Series,
) -> int | None:
    matching = game.loc[
        mask,
        "inning",
    ]

    if matching.empty:
        return None

    return int(
        matching.iloc[0]
    )


def _build_team_lead_row(
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

    margins = _team_margin(
        game,
        team,
    ).astype("int64")

    event_numbers = game[
        "scoring_event_number"
    ].astype("int64")

    ever_led = bool(
        margins.gt(0).any()
    )

    ever_led_by_2 = bool(
        margins.ge(2).any()
    )

    ever_led_by_3 = bool(
        margins.ge(3).any()
    )

    ever_led_by_4 = bool(
        margins.ge(4).any()
    )

    maximum_lead = int(
        max(
            0,
            int(margins.max()),
        )
    )

    maximum_deficit = int(
        max(
            0,
            int(-margins.min()),
        )
    )

    first_lead_event = _first_event_number(
        game,
        margins.gt(0),
    )

    first_two_run_lead_event = _first_event_number(
        game,
        margins.ge(2),
    )

    first_three_run_lead_event = _first_event_number(
        game,
        margins.ge(3),
    )

    first_lead_inning = _first_inning(
        game,
        margins.gt(0),
    )

    first_two_run_lead_inning = _first_inning(
        game,
        margins.ge(2),
    )

    first_three_run_lead_inning = _first_inning(
        game,
        margins.ge(3),
    )

    if first_lead_event is None:
        after_first_lead = pd.Series(
            dtype="int64"
        )

    else:
        after_first_lead = margins[
            event_numbers.ge(
                first_lead_event
            )
        ]

    if first_two_run_lead_event is None:
        after_first_two_run_lead = pd.Series(
            dtype="int64"
        )

    else:
        after_first_two_run_lead = margins[
            event_numbers.ge(
                first_two_run_lead_event
            )
        ]

    if first_three_run_lead_event is None:
        after_first_three_run_lead = pd.Series(
            dtype="int64"
        )

    else:
        after_first_three_run_lead = margins[
            event_numbers.ge(
                first_three_run_lead_event
            )
        ]

    tied_after_leading = bool(
        ever_led
        and after_first_lead.eq(0).any()
    )

    trailed_after_leading = bool(
        ever_led
        and after_first_lead.lt(0).any()
    )

    surrendered_lead = bool(
        tied_after_leading
        or trailed_after_leading
    )

    regained_lead_after_surrender = False

    if surrendered_lead:
        first_non_lead_index = after_first_lead[
            after_first_lead.le(0)
        ].index[0]

        regained_lead_after_surrender = bool(
            margins.loc[
                first_non_lead_index:
            ].gt(0).any()
        )

    final_margin = int(
        team_flow_row[
            "run_differential"
        ]
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

    covered_minus_1_5 = bool(
        final_margin >= 2
    )

    covered_plus_1_5 = bool(
        final_margin >= -1
    )

    two_run_lead_held_to_final = bool(
        ever_led_by_2
        and final_margin >= 2
    )

    three_run_lead_held_to_final = bool(
        ever_led_by_3
        and final_margin >= 3
    )

    led_by_2_but_failed_minus_1_5 = bool(
        ever_led_by_2
        and final_margin < 2
    )

    led_by_3_but_failed_minus_1_5 = bool(
        ever_led_by_3
        and final_margin < 2
    )

    led_but_lost = bool(
        ever_led
        and lost
    )

    led_by_2_but_lost = bool(
        ever_led_by_2
        and lost
    )

    led_by_3_but_lost = bool(
        ever_led_by_3
        and lost
    )

    winner_failed_to_separate = bool(
        won
        and final_margin == 1
    )

    winner_maintained_two_plus_after_first_two_run_lead = bool(
        won
        and ever_led_by_2
        and not after_first_two_run_lead.empty
        and after_first_two_run_lead.ge(2).all()
    )

    winner_maintained_three_plus_after_first_three_run_lead = bool(
        won
        and ever_led_by_3
        and not after_first_three_run_lead.empty
        and after_first_three_run_lead.ge(3).all()
    )

    dropped_below_two_after_reaching_two = bool(
        ever_led_by_2
        and after_first_two_run_lead.lt(2).any()
    )

    dropped_below_three_after_reaching_three = bool(
        ever_led_by_3
        and after_first_three_run_lead.lt(3).any()
    )

    late_two_run_lead_created = bool(
        first_two_run_lead_inning is not None
        and first_two_run_lead_inning >= 7
    )

    late_three_run_lead_created = bool(
        first_three_run_lead_inning is not None
        and first_three_run_lead_inning >= 7
    )

    final_margin_equals_maximum_lead = bool(
        won
        and final_margin == maximum_lead
    )

    gave_back_runs_after_maximum_lead = bool(
        won
        and maximum_lead > final_margin
    )

    decisive_score_for = bool(
        team_flow_row[
            "decisive_score_for"
        ]
    )

    decisive_lead_size = int(
        team_flow_row[
            "decisive_lead_size"
        ]
    )

    winner_additional_runs_after_decisive = int(
        team_flow_row[
            "winner_additional_runs_after_decisive"
        ]
    )

    winner_created_two_plus_after_decisive = bool(
        won
        and (
            decisive_lead_size >= 2
            or winner_additional_runs_after_decisive
            >= (
                2 - decisive_lead_size
            )
        )
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

        "final_run_differential":
            final_margin,

        "won":
            won,

        "lost":
            lost,

        "covered_minus_1_5":
            covered_minus_1_5,

        "covered_plus_1_5":
            covered_plus_1_5,

        "ever_led":
            ever_led,

        "ever_led_by_2":
            ever_led_by_2,

        "ever_led_by_3":
            ever_led_by_3,

        "ever_led_by_4":
            ever_led_by_4,

        "maximum_lead":
            maximum_lead,

        "maximum_deficit":
            maximum_deficit,

        "first_lead_event_number":
            first_lead_event,

        "first_lead_inning":
            first_lead_inning,

        "first_two_run_lead_event_number":
            first_two_run_lead_event,

        "first_two_run_lead_inning":
            first_two_run_lead_inning,

        "first_three_run_lead_event_number":
            first_three_run_lead_event,

        "first_three_run_lead_inning":
            first_three_run_lead_inning,

        "tied_after_leading":
            tied_after_leading,

        "trailed_after_leading":
            trailed_after_leading,

        "surrendered_lead":
            surrendered_lead,

        "regained_lead_after_surrender":
            regained_lead_after_surrender,

        "led_but_lost":
            led_but_lost,

        "led_by_2_but_lost":
            led_by_2_but_lost,

        "led_by_3_but_lost":
            led_by_3_but_lost,

        "two_run_lead_held_to_final":
            two_run_lead_held_to_final,

        "three_run_lead_held_to_final":
            three_run_lead_held_to_final,

        "led_by_2_but_failed_minus_1_5":
            led_by_2_but_failed_minus_1_5,

        "led_by_3_but_failed_minus_1_5":
            led_by_3_but_failed_minus_1_5,

        "dropped_below_two_after_reaching_two":
            dropped_below_two_after_reaching_two,

        "dropped_below_three_after_reaching_three":
            dropped_below_three_after_reaching_three,

        "winner_failed_to_separate":
            winner_failed_to_separate,

        "winner_maintained_two_plus_after_first_two_run_lead":
            winner_maintained_two_plus_after_first_two_run_lead,

        "winner_maintained_three_plus_after_first_three_run_lead":
            winner_maintained_three_plus_after_first_three_run_lead,

        "late_two_run_lead_created":
            late_two_run_lead_created,

        "late_three_run_lead_created":
            late_three_run_lead_created,

        "final_margin_equals_maximum_lead":
            final_margin_equals_maximum_lead,

        "gave_back_runs_after_maximum_lead":
            gave_back_runs_after_maximum_lead,

        "decisive_score_for":
            decisive_score_for,

        "decisive_lead_size":
            decisive_lead_size,

        "winner_additional_runs_after_decisive":
            winner_additional_runs_after_decisive,

        "winner_created_two_plus_after_decisive":
            winner_created_two_plus_after_decisive,

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

        "lead_protection_version":
            LEAD_PROTECTION_VERSION,
    }


def build_team_lead_protection(
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

    roles["atlas_season"] = pd.to_numeric(
        roles["atlas_season"],
        errors="raise",
    ).astype("int64")

    team_flow["game_pk"] = pd.to_numeric(
        team_flow["game_pk"],
        errors="raise",
    ).astype("int64")

    team_flow["atlas_season"] = pd.to_numeric(
        team_flow["atlas_season"],
        errors="raise",
    ).astype("int64")

    roles = roles[
        roles["atlas_season"].eq(
            int(season)
        )
    ].copy()

    team_flow = team_flow[
        team_flow["atlas_season"].eq(
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
            "Duplicate scoring-event role rows found."
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

    rows = []

    role_groups = {
        int(game_pk): game.copy()
        for game_pk, game
        in roles.groupby(
            "game_pk",
            sort=False,
        )
    }

    for team_row in team_flow.itertuples(
        index=False
    ):
        values = team_row._asdict()

        game_pk = int(
            values["game_pk"]
        )

        game = role_groups.get(
            game_pk
        )

        if game is None:
            raise KeyError(
                f"Missing scoring roles for game {game_pk}."
            )

        rows.append(
            _build_team_lead_row(
                game,
                pd.Series(values),
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


def audit_team_lead_protection(
    lead_facts: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows = []

    for game_pk, game in lead_facts.groupby(
        "game_pk",
        sort=False,
    ):
        exactly_two_rows = bool(
            len(game) == 2
        )

        one_winner = bool(
            int(
                game["won"].sum()
            ) == 1
        )

        one_loser = bool(
            int(
                game["lost"].sum()
            ) == 1
        )

        margin_mirror = bool(
            int(
                game[
                    "final_run_differential"
                ].sum()
            ) == 0
        )

        minus_line_consistent = bool(
            game[
                "covered_minus_1_5"
            ].eq(
                game[
                    "final_run_differential"
                ].ge(2)
            ).all()
        )

        two_run_hold_consistent = bool(
            game[
                "two_run_lead_held_to_final"
            ].eq(
                game["ever_led_by_2"]
                & game[
                    "final_run_differential"
                ].ge(2)
            ).all()
        )

        three_run_hold_consistent = bool(
            game[
                "three_run_lead_held_to_final"
            ].eq(
                game["ever_led_by_3"]
                & game[
                    "final_run_differential"
                ].ge(3)
            ).all()
        )

        led_but_lost_consistent = bool(
            game[
                "led_but_lost"
            ].eq(
                game["ever_led"]
                & game["lost"]
            ).all()
        )

        failed_separation_consistent = bool(
            game[
                "winner_failed_to_separate"
            ].eq(
                game["won"]
                & game[
                    "final_run_differential"
                ].eq(1)
            ).all()
        )

        no_negative_maximums = bool(
            game[
                "maximum_lead"
            ].ge(0).all()
            and game[
                "maximum_deficit"
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
            and margin_mirror
            and minus_line_consistent
            and two_run_hold_consistent
            and three_run_hold_consistent
            and led_but_lost_consistent
            and failed_separation_consistent
            and no_negative_maximums
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

            "margin_mirror":
                margin_mirror,

            "minus_line_consistent":
                minus_line_consistent,

            "two_run_hold_consistent":
                two_run_hold_consistent,

            "three_run_hold_consistent":
                three_run_hold_consistent,

            "led_but_lost_consistent":
                led_but_lost_consistent,

            "failed_separation_consistent":
                failed_separation_consistent,

            "no_negative_maximums":
                no_negative_maximums,

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


def run_lead_protection_build(
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

    lead_fact_path = (
        output_dir
        / "team_lead_protection.parquet"
    )

    audit_path = (
        output_dir
        / "team_lead_protection_audit.parquet"
    )

    failure_path = (
        output_dir
        / "team_lead_protection_failures.parquet"
    )

    metadata_path = (
        output_dir
        / "team_lead_protection_metadata.json"
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

        lead_facts = build_team_lead_protection(
            roles=roles,
            team_flow=team_flow,
            season=season,
        )

        audit = audit_team_lead_protection(
            lead_facts
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

        lead_facts = pd.DataFrame()
        audit = pd.DataFrame()

    duplicate_team_games = int(
        lead_facts.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
        if not lead_facts.empty
        else 0
    )

    audit_failures = (
        audit[
            ~audit["audit_pass"]
        ]
        if not audit.empty
        else pd.DataFrame()
    )

    games = int(
        lead_facts[
            "game_pk"
        ].nunique()
        if not lead_facts.empty
        else 0
    )

    teams = int(
        lead_facts[
            "team"
        ].nunique()
        if not lead_facts.empty
        else 0
    )

    phase_pass = bool(
        len(lead_facts) == 4_856
        and games == 2_428
        and teams == 30
        and failures.empty
        and audit_failures.empty
        and duplicate_team_games == 0
    )

    metadata = {
        "engine":
            "ATLAS Lead Protection and Separation Builder",

        "season":
            season,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "lead_protection_version":
            LEAD_PROTECTION_VERSION,

        "team_game_rows":
            int(
                len(lead_facts)
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

        "teams_ever_led":
            int(
                lead_facts[
                    "ever_led"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "teams_ever_led_by_2":
            int(
                lead_facts[
                    "ever_led_by_2"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "teams_ever_led_by_3":
            int(
                lead_facts[
                    "ever_led_by_3"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "led_but_lost":
            int(
                lead_facts[
                    "led_but_lost"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "led_by_2_but_lost":
            int(
                lead_facts[
                    "led_by_2_but_lost"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "led_by_2_but_failed_minus_1_5":
            int(
                lead_facts[
                    "led_by_2_but_failed_minus_1_5"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "two_run_leads_held_to_final":
            int(
                lead_facts[
                    "two_run_lead_held_to_final"
                ].sum()
                if not lead_facts.empty
                else 0
            ),

        "phase_2d4_pass":
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
            "lead_facts":
                str(lead_fact_path),

            "audit":
                str(audit_path),

            "failures":
                str(failure_path),

            "metadata":
                str(metadata_path),
        },
    }

    _atomic_parquet_write(
        lead_facts,
        lead_fact_path,
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
        "lead_facts":
            lead_facts,

        "audit":
            audit,

        "failures":
            failures,

        "metadata":
            metadata,
    }
