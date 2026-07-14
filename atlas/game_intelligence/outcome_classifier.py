"""
Deterministic factual game-outcome classification.

This module answers:

    What happened in the completed game?

It does not:

- explain why the outcome occurred
- update team or player identities
- create evidence or concepts
- create predictions
- use sportsbook information
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION
from .reconstruction import GameReconstruction


OUTCOME_CLASSIFIER_VERSION = "1.0.0"


@dataclass(frozen=True)
class GameOutcome:
    """
    One immutable factual outcome object per completed game.
    """

    game_pk: int
    game_date: pd.Timestamp
    atlas_season: int

    home_team: str
    away_team: str

    home_score: int
    away_score: int
    total_runs: int
    run_margin: int
    absolute_run_margin: int

    winner_side: str
    winner_team: str
    loser_team: str

    home_win: bool
    away_win: bool

    one_run_game: bool
    margin_gt_1_5: bool
    margin_gt_3_5: bool
    margin_gt_5_5: bool

    home_shutout_win: bool
    away_shutout_win: bool
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

    walkoff: bool
    walkoff_runs: int
    terminal_scoring_play: bool

    home_ever_led: bool
    away_ever_led: bool
    home_ever_trailed: bool
    away_ever_trailed: bool

    home_comeback_win: bool
    away_comeback_win: bool
    comeback_win: bool

    lead_changes: int
    times_tied_after_0_0: int

    largest_home_lead: int
    largest_away_lead: int
    largest_lead_any_team: int

    home_scored_first: bool
    away_scored_first: bool
    scoreless_game_through_3: bool
    scoreless_game_through_5: bool

    home_runs_innings_1_3: int
    away_runs_innings_1_3: int
    home_runs_innings_4_6: int
    away_runs_innings_4_6: int
    home_runs_innings_7_plus: int
    away_runs_innings_7_plus: int

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
    outcome_classifier_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric(
    values: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        values,
        errors="coerce",
    )


def _first_value(
    dataframe: pd.DataFrame,
    column: str,
) -> Any:
    if (
        dataframe.empty
        or column not in dataframe.columns
    ):
        return None

    values = dataframe[
        column
    ].dropna()

    if values.empty:
        return None

    return values.iloc[0]


def _normalize_half(
    value: Any,
) -> str:
    text = str(value).strip().lower()

    if text in {
        "top",
        "t",
    }:
        return "TOP"

    if text in {
        "bot",
        "bottom",
        "b",
    }:
        return "BOTTOM"

    return text.upper()


def _ordered_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    result = events.copy()

    result["_source_order"] = np.arange(
        len(result),
        dtype="int64",
    )

    sort_columns = [
        column
        for column in [
            "at_bat_number",
            "pitch_number",
            "_source_order",
        ]
        if column in result.columns
    ]

    if sort_columns:
        result = result.sort_values(
            sort_columns,
            kind="stable",
        )

    return result.reset_index(drop=True)


def _plate_appearance_ends(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the terminal event row for every plate appearance.
    """
    ordered = _ordered_events(
        events
    )

    if (
        ordered.empty
        or "at_bat_number"
        not in ordered.columns
    ):
        return ordered.copy()

    at_bat_number = _numeric(
        ordered["at_bat_number"]
    )

    ordered = ordered[
        at_bat_number.notna()
    ].copy()

    ordered[
        "at_bat_number"
    ] = at_bat_number[
        at_bat_number.notna()
    ].astype("int64")

    plate_appearances = (
        ordered.groupby(
            "at_bat_number",
            sort=True,
            as_index=False,
        )
        .tail(1)
        .sort_values(
            [
                "at_bat_number",
                "_source_order",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return plate_appearances


def _score_state_columns(
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    result = plate_appearances.copy()

    required = [
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
    ]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise KeyError(
            "Event sequence is missing score columns: "
            f"{missing}"
        )

    for column in required:
        result[column] = _numeric(
            result[column]
        )

    if result[
        required
    ].isna().any().any():
        raise ValueError(
            "Plate-appearance score states contain null values."
        )

    for column in required:
        result[column] = result[
            column
        ].astype("int64")

    result["home_runs_on_play"] = (
        result["post_home_score"]
        - result["home_score"]
    )

    result["away_runs_on_play"] = (
        result["post_away_score"]
        - result["away_score"]
    )

    result["runs_on_play"] = (
        result["home_runs_on_play"]
        + result["away_runs_on_play"]
    )

    result["post_run_margin"] = (
        result["post_home_score"]
        - result["post_away_score"]
    )

    result["post_leader"] = np.select(
        [
            result[
                "post_run_margin"
            ].gt(0),

            result[
                "post_run_margin"
            ].lt(0),
        ],
        [
            "HOME",
            "AWAY",
        ],
        default="TIE",
    )

    return result


def _lead_changes(
    leaders: pd.Series,
) -> int:
    """
    Count changes between home and away leaders.

    Tie states are ignored when identifying the prior leader.
    """
    non_ties = [
        str(value)
        for value in leaders
        if str(value) in {
            "HOME",
            "AWAY",
        }
    ]

    if len(non_ties) <= 1:
        return 0

    return int(
        sum(
            current != previous
            for previous, current
            in zip(
                non_ties,
                non_ties[1:],
            )
        )
    )


def _times_tied_after_zero(
    plate_appearances: pd.DataFrame,
) -> int:
    tied = (
        plate_appearances[
            "post_home_score"
        ].eq(
            plate_appearances[
                "post_away_score"
            ]
        )
        & plate_appearances[
            "post_home_score"
        ].gt(0)
    )

    if not tied.any():
        return 0

    # Count entries into a tied state rather than every pitch/PA
    # that remains tied.
    prior_tied = tied.shift(
        fill_value=False
    )

    return int(
        (
            tied
            & ~prior_tied
        ).sum()
    )


def _inning_scoring(
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    if "inning" not in plate_appearances.columns:
        raise KeyError(
            "Event sequence has no inning column."
        )

    frame = plate_appearances.copy()

    frame["inning"] = _numeric(
        frame["inning"]
    )

    if frame["inning"].isna().any():
        raise ValueError(
            "Plate appearances contain missing innings."
        )

    frame["inning"] = frame[
        "inning"
    ].astype("int64")

    scoring = (
        frame.groupby(
            "inning",
            sort=True,
        )
        .agg(
            home_runs=(
                "home_runs_on_play",
                "sum",
            ),
            away_runs=(
                "away_runs_on_play",
                "sum",
            ),
        )
        .reset_index()
    )

    scoring["total_runs"] = (
        scoring["home_runs"]
        + scoring["away_runs"]
    )

    return scoring


def _runs_in_range(
    inning_scoring: pd.DataFrame,
    score_column: str,
    minimum_inning: int,
    maximum_inning: int | None,
) -> int:
    mask = inning_scoring[
        "inning"
    ].ge(
        int(minimum_inning)
    )

    if maximum_inning is not None:
        mask &= inning_scoring[
            "inning"
        ].le(
            int(maximum_inning)
        )

    return int(
        inning_scoring.loc[
            mask,
            score_column,
        ].sum()
    )


def _score_after_inning(
    plate_appearances: pd.DataFrame,
    inning: int,
) -> tuple[int, int]:
    rows = plate_appearances[
        _numeric(
            plate_appearances["inning"]
        ).le(
            int(inning)
        )
    ]

    if rows.empty:
        return 0, 0

    final_row = rows.iloc[-1]

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


def _first_scoring_side(
    plate_appearances: pd.DataFrame,
) -> str | None:
    scoring = plate_appearances[
        plate_appearances[
            "runs_on_play"
        ].gt(0)
    ]

    if scoring.empty:
        return None

    first = scoring.iloc[0]

    if int(
        first[
            "home_runs_on_play"
        ]
    ) > 0:
        return "HOME"

    if int(
        first[
            "away_runs_on_play"
        ]
    ) > 0:
        return "AWAY"

    return None


def classify_game_outcome(
    reconstruction: GameReconstruction,
) -> GameOutcome:
    """
    Build one factual game-level outcome object.
    """
    if not reconstruction.validation.get(
        "reconstruction_pass",
        False,
    ):
        raise ValueError(
            "Only a fully verified reconstruction may be "
            "classified as a normal completed game."
        )

    core = reconstruction.game_core

    if len(core) != 1:
        raise ValueError(
            "Outcome classification requires one game-core row."
        )

    events = _ordered_events(
        reconstruction.events
    )

    plate_appearances = (
        _score_state_columns(
            _plate_appearance_ends(
                events
            )
        )
    )

    if plate_appearances.empty:
        raise ValueError(
            "Outcome classification requires plate appearances."
        )

    core_row = core.iloc[0]

    game_pk = int(
        core_row["game_pk"]
    )

    game_date = pd.Timestamp(
        core_row["game_date"]
    ).normalize()

    atlas_season = int(
        core_row["atlas_season"]
    )

    home_team = str(
        core_row["home_team"]
    )

    away_team = str(
        core_row["away_team"]
    )

    home_score = int(
        core_row["home_score"]
    )

    away_score = int(
        core_row["away_score"]
    )

    if home_score == away_score:
        raise AssertionError(
            f"Completed game {game_pk} has a tied final score."
        )

    total_runs = (
        home_score
        + away_score
    )

    run_margin = (
        home_score
        - away_score
    )

    absolute_run_margin = abs(
        run_margin
    )

    home_win = (
        home_score > away_score
    )

    away_win = (
        away_score > home_score
    )

    winner_side = (
        "HOME"
        if home_win
        else "AWAY"
    )

    winner_team = (
        home_team
        if home_win
        else away_team
    )

    loser_team = (
        away_team
        if home_win
        else home_team
    )

    final_pa = plate_appearances.iloc[-1]

    event_final_home = int(
        final_pa[
            "post_home_score"
        ]
    )

    event_final_away = int(
        final_pa[
            "post_away_score"
        ]
    )

    if (
        event_final_home != home_score
        or event_final_away != away_score
    ):
        raise AssertionError(
            f"Game {game_pk} core and terminal event scores disagree."
        )

    innings_played = int(
        _numeric(
            plate_appearances[
                "inning"
            ]
        ).max()
    )

    extra_innings = (
        innings_played > 9
    )

    score_after_9 = (
        _score_after_inning(
            plate_appearances,
            inning=9,
        )
    )

    tied_after_regulation = bool(
        extra_innings
        and score_after_9[0]
        == score_after_9[1]
    )

    final_half = _normalize_half(
        final_pa.get(
            "inning_topbot",
            "",
        )
    )

    final_home_runs = int(
        final_pa[
            "home_runs_on_play"
        ]
    )

    final_away_runs = int(
        final_pa[
            "away_runs_on_play"
        ]
    )

    terminal_scoring_play = bool(
        final_home_runs > 0
        or final_away_runs > 0
    )

    walkoff = bool(
        final_half == "BOTTOM"
        and home_win
        and final_home_runs > 0
        and int(
            final_pa[
                "home_score"
            ]
        ) <= int(
            final_pa[
                "away_score"
            ]
        )
        and home_score > away_score
    )

    walkoff_runs = (
        final_home_runs
        if walkoff
        else 0
    )

    post_margins = plate_appearances[
        "post_run_margin"
    ]

    home_ever_led = bool(
        post_margins.gt(0).any()
    )

    away_ever_led = bool(
        post_margins.lt(0).any()
    )

    home_ever_trailed = (
        away_ever_led
    )

    away_ever_trailed = (
        home_ever_led
    )

    home_comeback_win = bool(
        home_win
        and home_ever_trailed
    )

    away_comeback_win = bool(
        away_win
        and away_ever_trailed
    )

    comeback_win = bool(
        home_comeback_win
        or away_comeback_win
    )

    largest_home_lead = int(
        max(
            0,
            post_margins.max(),
        )
    )

    largest_away_lead = int(
        max(
            0,
            -post_margins.min(),
        )
    )

    largest_lead_any_team = max(
        largest_home_lead,
        largest_away_lead,
    )

    lead_changes = _lead_changes(
        plate_appearances[
            "post_leader"
        ]
    )

    times_tied_after_0_0 = (
        _times_tied_after_zero(
            plate_appearances
        )
    )

    first_scoring_side = (
        _first_scoring_side(
            plate_appearances
        )
    )

    home_scored_first = (
        first_scoring_side == "HOME"
    )

    away_scored_first = (
        first_scoring_side == "AWAY"
    )

    score_after_3 = (
        _score_after_inning(
            plate_appearances,
            inning=3,
        )
    )

    score_after_5 = (
        _score_after_inning(
            plate_appearances,
            inning=5,
        )
    )

    scoreless_game_through_3 = (
        score_after_3 == (0, 0)
    )

    scoreless_game_through_5 = (
        score_after_5 == (0, 0)
    )

    inning_scoring = (
        _inning_scoring(
            plate_appearances
        )
    )

    home_runs_innings_1_3 = (
        _runs_in_range(
            inning_scoring,
            "home_runs",
            1,
            3,
        )
    )

    away_runs_innings_1_3 = (
        _runs_in_range(
            inning_scoring,
            "away_runs",
            1,
            3,
        )
    )

    home_runs_innings_4_6 = (
        _runs_in_range(
            inning_scoring,
            "home_runs",
            4,
            6,
        )
    )

    away_runs_innings_4_6 = (
        _runs_in_range(
            inning_scoring,
            "away_runs",
            4,
            6,
        )
    )

    home_runs_innings_7_plus = (
        _runs_in_range(
            inning_scoring,
            "home_runs",
            7,
            None,
        )
    )

    away_runs_innings_7_plus = (
        _runs_in_range(
            inning_scoring,
            "away_runs",
            7,
            None,
        )
    )

    late_runs = (
        home_runs_innings_7_plus
        + away_runs_innings_7_plus
    )

    late_scoring_game = bool(
        late_runs > 0
    )

    final_inning_scoring = bool(
        int(
            inning_scoring.loc[
                inning_scoring[
                    "inning"
                ].eq(
                    innings_played
                ),
                "total_runs",
            ].sum()
        ) > 0
    )

    scoring_plays = int(
        plate_appearances[
            "runs_on_play"
        ].gt(0).sum()
    )

    return GameOutcome(
        game_pk=game_pk,
        game_date=game_date,
        atlas_season=atlas_season,

        home_team=home_team,
        away_team=away_team,

        home_score=home_score,
        away_score=away_score,
        total_runs=total_runs,
        run_margin=run_margin,
        absolute_run_margin=absolute_run_margin,

        winner_side=winner_side,
        winner_team=winner_team,
        loser_team=loser_team,

        home_win=home_win,
        away_win=away_win,

        one_run_game=(
            absolute_run_margin == 1
        ),
        margin_gt_1_5=(
            absolute_run_margin >= 2
        ),
        margin_gt_3_5=(
            absolute_run_margin >= 4
        ),
        margin_gt_5_5=(
            absolute_run_margin >= 6
        ),

        home_shutout_win=bool(
            home_win
            and away_score == 0
        ),
        away_shutout_win=bool(
            away_win
            and home_score == 0
        ),
        either_team_shut_out=bool(
            home_score == 0
            or away_score == 0
        ),

        game_total_5_or_less=(
            total_runs <= 5
        ),
        game_total_6_or_less=(
            total_runs <= 6
        ),
        game_total_7_or_less=(
            total_runs <= 7
        ),
        game_total_8_or_less=(
            total_runs <= 8
        ),
        game_total_9_plus=(
            total_runs >= 9
        ),
        game_total_10_plus=(
            total_runs >= 10
        ),
        game_total_11_plus=(
            total_runs >= 11
        ),
        game_total_12_plus=(
            total_runs >= 12
        ),
        game_total_15_plus=(
            total_runs >= 15
        ),
        game_total_17_plus=(
            total_runs >= 17
        ),

        innings_played=innings_played,
        extra_innings=extra_innings,
        tied_after_regulation=(
            tied_after_regulation
        ),

        walkoff=walkoff,
        walkoff_runs=walkoff_runs,
        terminal_scoring_play=(
            terminal_scoring_play
        ),

        home_ever_led=home_ever_led,
        away_ever_led=away_ever_led,
        home_ever_trailed=(
            home_ever_trailed
        ),
        away_ever_trailed=(
            away_ever_trailed
        ),

        home_comeback_win=(
            home_comeback_win
        ),
        away_comeback_win=(
            away_comeback_win
        ),
        comeback_win=comeback_win,

        lead_changes=lead_changes,
        times_tied_after_0_0=(
            times_tied_after_0_0
        ),

        largest_home_lead=(
            largest_home_lead
        ),
        largest_away_lead=(
            largest_away_lead
        ),
        largest_lead_any_team=(
            largest_lead_any_team
        ),

        home_scored_first=(
            home_scored_first
        ),
        away_scored_first=(
            away_scored_first
        ),
        scoreless_game_through_3=(
            scoreless_game_through_3
        ),
        scoreless_game_through_5=(
            scoreless_game_through_5
        ),

        home_runs_innings_1_3=(
            home_runs_innings_1_3
        ),
        away_runs_innings_1_3=(
            away_runs_innings_1_3
        ),
        home_runs_innings_4_6=(
            home_runs_innings_4_6
        ),
        away_runs_innings_4_6=(
            away_runs_innings_4_6
        ),
        home_runs_innings_7_plus=(
            home_runs_innings_7_plus
        ),
        away_runs_innings_7_plus=(
            away_runs_innings_7_plus
        ),

        late_scoring_game=(
            late_scoring_game
        ),
        final_inning_scoring=(
            final_inning_scoring
        ),

        scoring_plays=scoring_plays,
        event_rows=int(
            len(events)
        ),
        plate_appearances=int(
            len(plate_appearances)
        ),

        score_sources_verified=bool(
            reconstruction.validation.get(
                "scores_agree",
                False,
            )
        ),
        reconstruction_verified=bool(
            reconstruction.validation.get(
                "reconstruction_pass",
                False,
            )
        ),

        prediction_created=False,
        identity_updated=False,
        explanation_created=False,
        future_games_used=False,

        brain_engine_version=(
            BRAIN_ENGINE_VERSION
        ),
        outcome_classifier_version=(
            OUTCOME_CLASSIFIER_VERSION
        ),
    )


def outcome_to_frame(
    outcome: GameOutcome,
) -> pd.DataFrame:
    """
    Convert one outcome object to a one-row DataFrame.
    """
    return pd.DataFrame(
        [
            outcome.to_dict()
        ]
    )
