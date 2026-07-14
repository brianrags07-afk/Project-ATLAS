"""
Team-perspective factual outcome classification.

This module converts one verified game-level outcome into exactly
two factual rows:

- one home-team perspective
- one away-team perspective

It does not:

- explain why the game happened
- update team or player identities
- discover evidence or concepts
- create predictions
- use sportsbook information
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION
from .outcome_classifier import GameOutcome


TEAM_OUTCOME_CLASSIFIER_VERSION = "1.0.0"


@dataclass(frozen=True)
class TeamGameOutcome:
    """
    One immutable factual outcome from one team's perspective.
    """

    game_pk: int
    game_date: pd.Timestamp
    atlas_season: int

    team: str
    opponent: str
    home_away: str

    team_score: int
    opponent_score: int
    total_runs: int

    run_differential: int
    absolute_run_margin: int

    won: bool
    lost: bool

    winner_team: str
    loser_team: str

    one_run_game: bool

    won_by_1: bool
    won_by_2_plus: bool
    won_by_4_plus: bool
    won_by_6_plus: bool

    lost_by_1: bool
    lost_by_2_plus: bool
    lost_by_4_plus: bool
    lost_by_6_plus: bool

    covered_minus_1_5_result: bool
    covered_minus_3_5_result: bool
    covered_minus_5_5_result: bool

    lost_plus_1_5_result: bool
    lost_plus_3_5_result: bool
    lost_plus_5_5_result: bool

    team_scored_0: bool
    team_scored_1_or_less: bool
    team_scored_2_or_less: bool
    team_scored_3_or_less: bool
    team_scored_exactly_4: bool
    team_scored_5_plus: bool
    team_scored_6_plus: bool
    team_scored_8_plus: bool
    team_scored_10_plus: bool

    team_allowed_0: bool
    team_allowed_1_or_less: bool
    team_allowed_2_or_less: bool
    team_allowed_3_or_less: bool
    team_allowed_exactly_4: bool
    team_allowed_5_plus: bool
    team_allowed_6_plus: bool
    team_allowed_8_plus: bool
    team_allowed_10_plus: bool

    shutout_win: bool
    shutout_loss: bool
    either_team_shut_out: bool

    game_total_5_or_less: bool
    game_total_6_or_less: bool
    game_total_7_or_less: bool
    game_total_8_or_less: bool
    game_total_9_plus: bool
    game_total_10_plus: bool
    game_total_11_plus: bool
    game_total_12_plus: bool
    game_total_15_plus: bool
    game_total_17_plus: bool

    innings_played: int
    extra_innings: bool
    tied_after_regulation: bool

    walkoff_game: bool
    walkoff_win: bool
    walkoff_loss: bool
    walkoff_runs: int

    team_ever_led: bool
    opponent_ever_led: bool
    team_ever_trailed: bool
    opponent_ever_trailed: bool

    comeback_win: bool
    comeback_loss: bool

    lead_changes: int
    times_tied_after_0_0: int

    largest_team_lead: int
    largest_opponent_lead: int
    largest_deficit_overcome: int
    largest_lead_lost: int

    team_scored_first: bool
    opponent_scored_first: bool

    scoreless_game_through_3: bool
    scoreless_game_through_5: bool

    team_runs_innings_1_3: int
    opponent_runs_innings_1_3: int

    team_runs_innings_4_6: int
    opponent_runs_innings_4_6: int

    team_runs_innings_7_plus: int
    opponent_runs_innings_7_plus: int

    team_scored_late: bool
    opponent_scored_late: bool
    late_scoring_game: bool
    final_inning_scoring: bool

    scoring_plays: int
    event_rows: int
    plate_appearances: int

    score_sources_verified: bool
    reconstruction_verified: bool

    prediction_created: bool
    identity_updated: bool
    explanation_created: bool
    future_games_used: bool

    brain_engine_version: str
    team_outcome_classifier_version: str
    source_outcome_classifier_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_team_row(
    outcome: GameOutcome,
    side: str,
) -> TeamGameOutcome:
    side = str(side).upper()

    if side not in {
        "HOME",
        "AWAY",
    }:
        raise ValueError(
            f"Unknown team perspective: {side}"
        )

    is_home = side == "HOME"

    team = (
        outcome.home_team
        if is_home
        else outcome.away_team
    )

    opponent = (
        outcome.away_team
        if is_home
        else outcome.home_team
    )

    team_score = (
        outcome.home_score
        if is_home
        else outcome.away_score
    )

    opponent_score = (
        outcome.away_score
        if is_home
        else outcome.home_score
    )

    won = (
        outcome.home_win
        if is_home
        else outcome.away_win
    )

    lost = not won

    run_differential = (
        team_score
        - opponent_score
    )

    team_ever_led = (
        outcome.home_ever_led
        if is_home
        else outcome.away_ever_led
    )

    opponent_ever_led = (
        outcome.away_ever_led
        if is_home
        else outcome.home_ever_led
    )

    team_ever_trailed = (
        outcome.home_ever_trailed
        if is_home
        else outcome.away_ever_trailed
    )

    opponent_ever_trailed = (
        outcome.away_ever_trailed
        if is_home
        else outcome.home_ever_trailed
    )

    comeback_win = (
        outcome.home_comeback_win
        if is_home
        else outcome.away_comeback_win
    )

    comeback_loss = bool(
        lost
        and opponent_ever_trailed
    )

    largest_team_lead = (
        outcome.largest_home_lead
        if is_home
        else outcome.largest_away_lead
    )

    largest_opponent_lead = (
        outcome.largest_away_lead
        if is_home
        else outcome.largest_home_lead
    )

    largest_deficit_overcome = (
        largest_opponent_lead
        if comeback_win
        else 0
    )

    largest_lead_lost = (
        largest_team_lead
        if comeback_loss
        else 0
    )

    team_scored_first = (
        outcome.home_scored_first
        if is_home
        else outcome.away_scored_first
    )

    opponent_scored_first = (
        outcome.away_scored_first
        if is_home
        else outcome.home_scored_first
    )

    team_runs_innings_1_3 = (
        outcome.home_runs_innings_1_3
        if is_home
        else outcome.away_runs_innings_1_3
    )

    opponent_runs_innings_1_3 = (
        outcome.away_runs_innings_1_3
        if is_home
        else outcome.home_runs_innings_1_3
    )

    team_runs_innings_4_6 = (
        outcome.home_runs_innings_4_6
        if is_home
        else outcome.away_runs_innings_4_6
    )

    opponent_runs_innings_4_6 = (
        outcome.away_runs_innings_4_6
        if is_home
        else outcome.home_runs_innings_4_6
    )

    team_runs_innings_7_plus = (
        outcome.home_runs_innings_7_plus
        if is_home
        else outcome.away_runs_innings_7_plus
    )

    opponent_runs_innings_7_plus = (
        outcome.away_runs_innings_7_plus
        if is_home
        else outcome.home_runs_innings_7_plus
    )

    walkoff_win = bool(
        is_home
        and outcome.walkoff
        and won
    )

    walkoff_loss = bool(
        not is_home
        and outcome.walkoff
        and lost
    )

    return TeamGameOutcome(
        game_pk=outcome.game_pk,
        game_date=outcome.game_date,
        atlas_season=outcome.atlas_season,

        team=team,
        opponent=opponent,
        home_away=side,

        team_score=team_score,
        opponent_score=opponent_score,
        total_runs=outcome.total_runs,

        run_differential=run_differential,
        absolute_run_margin=(
            outcome.absolute_run_margin
        ),

        won=won,
        lost=lost,

        winner_team=outcome.winner_team,
        loser_team=outcome.loser_team,

        one_run_game=outcome.one_run_game,

        won_by_1=bool(
            won
            and run_differential == 1
        ),
        won_by_2_plus=bool(
            won
            and run_differential >= 2
        ),
        won_by_4_plus=bool(
            won
            and run_differential >= 4
        ),
        won_by_6_plus=bool(
            won
            and run_differential >= 6
        ),

        lost_by_1=bool(
            lost
            and run_differential == -1
        ),
        lost_by_2_plus=bool(
            lost
            and run_differential <= -2
        ),
        lost_by_4_plus=bool(
            lost
            and run_differential <= -4
        ),
        lost_by_6_plus=bool(
            lost
            and run_differential <= -6
        ),

        covered_minus_1_5_result=bool(
            run_differential >= 2
        ),
        covered_minus_3_5_result=bool(
            run_differential >= 4
        ),
        covered_minus_5_5_result=bool(
            run_differential >= 6
        ),

        lost_plus_1_5_result=bool(
            run_differential <= -2
        ),
        lost_plus_3_5_result=bool(
            run_differential <= -4
        ),
        lost_plus_5_5_result=bool(
            run_differential <= -6
        ),

        team_scored_0=(
            team_score == 0
        ),
        team_scored_1_or_less=(
            team_score <= 1
        ),
        team_scored_2_or_less=(
            team_score <= 2
        ),
        team_scored_3_or_less=(
            team_score <= 3
        ),
        team_scored_exactly_4=(
            team_score == 4
        ),
        team_scored_5_plus=(
            team_score >= 5
        ),
        team_scored_6_plus=(
            team_score >= 6
        ),
        team_scored_8_plus=(
            team_score >= 8
        ),
        team_scored_10_plus=(
            team_score >= 10
        ),

        team_allowed_0=(
            opponent_score == 0
        ),
        team_allowed_1_or_less=(
            opponent_score <= 1
        ),
        team_allowed_2_or_less=(
            opponent_score <= 2
        ),
        team_allowed_3_or_less=(
            opponent_score <= 3
        ),
        team_allowed_exactly_4=(
            opponent_score == 4
        ),
        team_allowed_5_plus=(
            opponent_score >= 5
        ),
        team_allowed_6_plus=(
            opponent_score >= 6
        ),
        team_allowed_8_plus=(
            opponent_score >= 8
        ),
        team_allowed_10_plus=(
            opponent_score >= 10
        ),

        shutout_win=bool(
            won
            and opponent_score == 0
        ),
        shutout_loss=bool(
            lost
            and team_score == 0
        ),
        either_team_shut_out=(
            outcome.either_team_shut_out
        ),

        game_total_5_or_less=(
            outcome.game_total_5_or_less
        ),
        game_total_6_or_less=(
            outcome.game_total_6_or_less
        ),
        game_total_7_or_less=(
            outcome.game_total_7_or_less
        ),
        game_total_8_or_less=(
            outcome.game_total_8_or_less
        ),
        game_total_9_plus=(
            outcome.game_total_9_plus
        ),
        game_total_10_plus=(
            outcome.game_total_10_plus
        ),
        game_total_11_plus=(
            outcome.game_total_11_plus
        ),
        game_total_12_plus=(
            outcome.game_total_12_plus
        ),
        game_total_15_plus=(
            outcome.game_total_15_plus
        ),
        game_total_17_plus=(
            outcome.game_total_17_plus
        ),

        innings_played=(
            outcome.innings_played
        ),
        extra_innings=(
            outcome.extra_innings
        ),
        tied_after_regulation=(
            outcome.tied_after_regulation
        ),

        walkoff_game=outcome.walkoff,
        walkoff_win=walkoff_win,
        walkoff_loss=walkoff_loss,
        walkoff_runs=(
            outcome.walkoff_runs
        ),

        team_ever_led=team_ever_led,
        opponent_ever_led=opponent_ever_led,
        team_ever_trailed=(
            team_ever_trailed
        ),
        opponent_ever_trailed=(
            opponent_ever_trailed
        ),

        comeback_win=comeback_win,
        comeback_loss=comeback_loss,

        lead_changes=outcome.lead_changes,
        times_tied_after_0_0=(
            outcome.times_tied_after_0_0
        ),

        largest_team_lead=(
            largest_team_lead
        ),
        largest_opponent_lead=(
            largest_opponent_lead
        ),
        largest_deficit_overcome=(
            largest_deficit_overcome
        ),
        largest_lead_lost=(
            largest_lead_lost
        ),

        team_scored_first=(
            team_scored_first
        ),
        opponent_scored_first=(
            opponent_scored_first
        ),

        scoreless_game_through_3=(
            outcome.scoreless_game_through_3
        ),
        scoreless_game_through_5=(
            outcome.scoreless_game_through_5
        ),

        team_runs_innings_1_3=(
            team_runs_innings_1_3
        ),
        opponent_runs_innings_1_3=(
            opponent_runs_innings_1_3
        ),

        team_runs_innings_4_6=(
            team_runs_innings_4_6
        ),
        opponent_runs_innings_4_6=(
            opponent_runs_innings_4_6
        ),

        team_runs_innings_7_plus=(
            team_runs_innings_7_plus
        ),
        opponent_runs_innings_7_plus=(
            opponent_runs_innings_7_plus
        ),

        team_scored_late=bool(
            team_runs_innings_7_plus > 0
        ),
        opponent_scored_late=bool(
            opponent_runs_innings_7_plus > 0
        ),
        late_scoring_game=(
            outcome.late_scoring_game
        ),
        final_inning_scoring=(
            outcome.final_inning_scoring
        ),

        scoring_plays=outcome.scoring_plays,
        event_rows=outcome.event_rows,
        plate_appearances=(
            outcome.plate_appearances
        ),

        score_sources_verified=(
            outcome.score_sources_verified
        ),
        reconstruction_verified=(
            outcome.reconstruction_verified
        ),

        prediction_created=False,
        identity_updated=False,
        explanation_created=False,
        future_games_used=False,

        brain_engine_version=(
            BRAIN_ENGINE_VERSION
        ),
        team_outcome_classifier_version=(
            TEAM_OUTCOME_CLASSIFIER_VERSION
        ),
        source_outcome_classifier_version=(
            outcome.outcome_classifier_version
        ),
    )


def classify_team_outcomes(
    outcome: GameOutcome,
) -> tuple[
    TeamGameOutcome,
    TeamGameOutcome,
]:
    """
    Convert one game-level outcome into home and away perspectives.
    """
    if not outcome.reconstruction_verified:
        raise ValueError(
            "Team outcomes require a verified reconstruction."
        )

    if not outcome.score_sources_verified:
        raise ValueError(
            "Team outcomes require verified score sources."
        )

    home = _build_team_row(
        outcome=outcome,
        side="HOME",
    )

    away = _build_team_row(
        outcome=outcome,
        side="AWAY",
    )

    if home.team != outcome.home_team:
        raise AssertionError(
            "Home perspective team mismatch."
        )

    if away.team != outcome.away_team:
        raise AssertionError(
            "Away perspective team mismatch."
        )

    if home.opponent != away.team:
        raise AssertionError(
            "Home opponent does not match away team."
        )

    if away.opponent != home.team:
        raise AssertionError(
            "Away opponent does not match home team."
        )

    if int(home.won) + int(away.won) != 1:
        raise AssertionError(
            "Exactly one team perspective must be a win."
        )

    if int(home.lost) + int(away.lost) != 1:
        raise AssertionError(
            "Exactly one team perspective must be a loss."
        )

    if (
        home.run_differential
        != -away.run_differential
    ):
        raise AssertionError(
            "Team run differentials must be opposites."
        )

    return home, away


def team_outcomes_to_frame(
    outcomes: tuple[
        TeamGameOutcome,
        TeamGameOutcome,
    ],
) -> pd.DataFrame:
    """
    Convert both team perspectives to a two-row DataFrame.
    """
    return pd.DataFrame(
        [
            outcome.to_dict()
            for outcome in outcomes
        ]
    ).sort_values(
        "home_away",
        kind="stable",
    ).reset_index(drop=True)
