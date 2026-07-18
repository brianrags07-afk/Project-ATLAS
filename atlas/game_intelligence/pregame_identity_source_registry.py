"""
Phase 2E.1 pregame identity source registry builder for Project ATLAS.

This module governs the frozen Phase 2D team-game fact columns
(``atlas_reference/schemas/data__game_intelligence__game_flow_facts__2024__
team_game_flow_facts.parquet.schema.json``) and emits the deterministic
governance registry consumed by Phase 2E.2 (strict prior-date team identity
timeline).

Contract source
----------------

The column classification embedded in this module is not a heuristic. It is
transcribed directly from the authoritative registry contract shipped in the
repository at:

- ``atlas_reference/schemas/data__game_intelligence__pregame_identity_registry
  __2024__pregame_identity_source_registry.csv.schema.json``
- ``atlas_reference/samples/general/data__game_intelligence__
  pregame_identity_registry__2024__pregame_identity_source_registry.csv.
  sample.parquet``

Every one of the 121 frozen Phase 2D columns has an explicit governance
decision (``identity_key``, ``lagged_identity_source``, or ``exclude``).
Columns encountered outside this contract are rejected rather than guessed,
and contract columns missing from an input frame are also rejected, per the
ATLAS non-negotiable: never invent column names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import json

import pandas as pd

from atlas.config import DATA_ROOT


ENGINE_VERSION: Final[str] = "1.0.0"

JOIN_KEYS: Final[tuple[str, ...]] = (
    "game_pk",
    "team",
)

EXPECTED_TOTAL_COLUMNS: Final[int] = 121
EXPECTED_LAGGED_SOURCE_COUNT: Final[int] = 87

# column -> (family, source_status, same_game_safe, requires_shift, historical_aggregation_allowed)
# Transcribed verbatim, in the authoritative registry's own row order, from
# the frozen contract described in the module docstring. This governance
# ordering (grouped by family, not by the raw Phase 2D source column order)
# is itself part of the frozen contract and is reproduced exactly.
IDENTITY_SOURCE_CLASSIFICATION: Final[dict[str, tuple[str, str, bool, bool, bool]]] = {
    "loser_margin_match": ("audit", "exclude", False, False, False),
    "minus_1_5_math_match": ("audit", "exclude", False, False, False),
    "plus_1_5_math_match": ("audit", "exclude", False, False, False),
    "role_decisive_events": ("audit", "exclude", False, False, False),
    "role_go_ahead_scores": ("audit", "exclude", False, False, False),
    "role_late_scoring_events": ("audit", "exclude", False, False, False),
    "role_opening_scores": ("audit", "exclude", False, False, False),
    "role_scoring_events": ("audit", "exclude", False, False, False),
    "role_tying_scores": ("audit", "exclude", False, False, False),
    "score_math_match": ("audit", "exclude", False, False, False),
    "shared_game_date_match": ("audit", "exclude", False, False, False),
    "shared_home_away_match": ("audit", "exclude", False, False, False),
    "shared_margin_match": ("audit", "exclude", False, False, False),
    "shared_opponent_match": ("audit", "exclude", False, False, False),
    "shared_run_line_match": ("audit", "exclude", False, False, False),
    "shared_score_match": ("audit", "exclude", False, False, False),
    "shared_season_match": ("audit", "exclude", False, False, False),
    "shared_win_loss_match": ("audit", "exclude", False, False, False),
    "winner_margin_match": ("audit", "exclude", False, False, False),
    "all_cross_layer_checks_pass": ("provenance", "exclude", False, False, False),
    "brain_engine_version": ("provenance", "exclude", False, False, False),
    "explanation_created": ("provenance", "exclude", False, False, False),
    "future_games_used": ("provenance", "exclude", False, False, False),
    "game_flow_fact_table_version": ("provenance", "exclude", False, False, False),
    "identity_updated": ("provenance", "exclude", False, False, False),
    "postgame_factual_only": ("provenance", "exclude", False, False, False),
    "prediction_created": ("provenance", "exclude", False, False, False),
    "pregame_feature_safe": ("provenance", "exclude", False, False, False),
    "atlas_season": ("identity", "identity_key", True, False, False),
    "game_date": ("identity", "identity_key", True, False, False),
    "game_pk": ("identity", "identity_key", True, False, False),
    "home_away": ("identity", "identity_key", True, False, False),
    "opponent": ("identity", "identity_key", True, False, False),
    "team": ("identity", "identity_key", True, False, False),
    "flow__allowed_first_score": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__decisive_inning": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__decisive_lead_size": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__decisive_score_against": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__decisive_score_for": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__failed_minus_1_5_as_winner": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__lost_by_2_plus": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__lost_by_3_plus": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__scored_first": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_deficit_reductions": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_go_ahead_scores": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_late_lead_extensions": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_late_runs": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_late_scoring_events": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_lead_extensions": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_tying_scores": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__winner_additional_runs_after_decisive": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__won_by_2_plus": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__won_by_3_plus": ("game_flow", "lagged_identity_source", False, True, True),
    "lead__dropped_below_two_after_reaching_two": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__ever_led": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__ever_led_by_2": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__ever_led_by_3": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__ever_led_by_4": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__first_lead_inning": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__first_three_run_lead_inning": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__first_two_run_lead_inning": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__gave_back_runs_after_maximum_lead": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__led_but_lost": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__led_by_2_but_failed_minus_1_5": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__led_by_2_but_lost": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__led_by_3_but_failed_minus_1_5": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__led_by_3_but_lost": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__maximum_deficit": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__maximum_lead": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__regained_lead_after_surrender": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__surrendered_lead": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__three_run_lead_held_to_final": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__two_run_lead_held_to_final": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__winner_created_two_plus_after_decisive": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__winner_failed_to_separate": ("lead_protection", "lagged_identity_source", False, True, True),
    "lead__winner_maintained_two_plus_after_first_two_run_lead": ("lead_protection", "lagged_identity_source", False, True, True),
    "covered_minus_1_5": ("other", "lagged_identity_source", False, True, True),
    "covered_plus_1_5": ("other", "lagged_identity_source", False, True, True),
    "lost": ("other", "lagged_identity_source", False, True, True),
    "opponent_score": ("other", "lagged_identity_source", False, True, True),
    "run_differential": ("other", "lagged_identity_source", False, True, True),
    "team_score": ("other", "lagged_identity_source", False, True, True),
    "won": ("other", "lagged_identity_source", False, True, True),
    "response__answered_after_falling_behind": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__average_response_event_gap": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__average_response_inning_gap": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__eventual_response_rate": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__go_ahead_responses": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__immediate_response_rate": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__late_responses": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__longest_opponent_unanswered_event_streak": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__longest_opponent_unanswered_run_streak": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__lost_after_scoring_first": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__opponent_scoring_events": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__same_inning_responses": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__team_eventually_responded": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__team_scored_next_after_opponent_event": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__tied_after_falling_behind": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__took_lead_after_falling_behind": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__tying_responses": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__unanswered_opponent_scoring_events": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__within_one_inning_responses": ("response_recovery", "lagged_identity_source", False, True, True),
    "response__won_after_allowing_first_score": ("response_recovery", "lagged_identity_source", False, True, True),
    "role_deficit_reductions": ("scoring_role_summary", "lagged_identity_source", False, True, True),
    "role_lead_extensions": ("scoring_role_summary", "lagged_identity_source", False, True, True),
    "outcome__comeback_loss": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__comeback_win": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__largest_deficit_overcome": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__largest_lead_lost": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__loss_by_2_plus": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__loss_by_4_plus": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__one_run_loss": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__one_run_win": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__shutout_loss": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__shutout_win": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__walkoff_loss": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__walkoff_win": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__win_by_2_plus": ("team_outcome", "lagged_identity_source", False, True, True),
    "outcome__win_by_4_plus": ("team_outcome", "lagged_identity_source", False, True, True),
    "flow__opponent_scoring_events": ("game_flow", "lagged_identity_source", False, True, True),
    "flow__team_scoring_events": ("game_flow", "lagged_identity_source", False, True, True),
}


def _classification_reason(
    source_status: str,
    family: str,
    column: str,
) -> str:
    if source_status == "identity_key":
        return "Used only for ordering, grouping or joining."

    if source_status == "exclude":
        if family == "provenance":
            return "Provenance/control field; not a baseball identity metric."
        return "Audit or validation field; not a predictive baseball fact."

    if column == "flow__opponent_scoring_events":
        return "Historical opponent scoring-event count; safe only after prior-date lag."

    if column == "flow__team_scoring_events":
        return "Historical scoring-event count; safe only after prior-date lag."

    return "Postgame fact permitted only after strict chronological lag."


def normalize_phase_2d_identity_inputs(
    dataframe: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError(
            "Phase 2D identity source dataframe is empty."
        )

    missing = sorted(
        set(JOIN_KEYS).difference(
            dataframe.columns
        )
    )
    if missing:
        raise KeyError(
            f"Missing required team-game keys: {missing}"
        )

    normalized = dataframe.copy()
    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["team"] = (
        normalized["team"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(
            normalized["game_date"],
            errors="raise",
        ).dt.normalize()

    if "atlas_season" not in normalized.columns:
        normalized["atlas_season"] = int(season)

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    wrong_season = normalized[
        "atlas_season"
    ].ne(int(season))
    if wrong_season.any():
        raise AssertionError(
            f"Phase 2D inputs contain rows outside season {season}."
        )

    duplicate_count = int(
        normalized.duplicated(
            subset=list(JOIN_KEYS),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise AssertionError(
            f"Phase 2D inputs contain duplicate team-game rows: {duplicate_count:,}"
        )

    return normalized.reset_index(
        drop=True
    )


def assert_matches_frozen_contract(
    dataframe: pd.DataFrame,
) -> None:
    """Reject columns outside the frozen Phase 2D contract and reject a
    frame that is missing any frozen Phase 2D contract column.

    This is the concrete enforcement of the ATLAS non-negotiable
    "never invent column names": the registry never guesses a
    classification for an unknown column, and never silently proceeds
    with part of the frozen contract missing.
    """
    frame_columns = set(dataframe.columns)
    contract_columns = set(IDENTITY_SOURCE_CLASSIFICATION)

    unknown_columns = sorted(
        frame_columns.difference(
            contract_columns
        )
    )
    if unknown_columns:
        raise KeyError(
            "Input frame contains columns outside the frozen Phase 2D "
            f"identity source contract: {unknown_columns}"
        )

    missing_columns = sorted(
        contract_columns.difference(
            frame_columns
        )
    )
    if missing_columns:
        raise KeyError(
            "Input frame is missing frozen Phase 2D identity source "
            f"contract columns: {missing_columns}"
        )


def build_pregame_identity_source_registry(
    phase_2d_identity_frame: pd.DataFrame,
    season: int,
    expected_total_columns: int = EXPECTED_TOTAL_COLUMNS,
    expected_source_count: int = EXPECTED_LAGGED_SOURCE_COUNT,
) -> pd.DataFrame:
    normalized = normalize_phase_2d_identity_inputs(
        phase_2d_identity_frame,
        season=season,
    )

    assert_matches_frozen_contract(
        normalized
    )

    rows = []
    for column, classification in IDENTITY_SOURCE_CLASSIFICATION.items():
        family, source_status, same_game_safe, requires_shift, historical_aggregation_allowed = (
            classification
        )

        series = normalized[column]

        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "family": family,
            "source_status": source_status,
            "same_game_safe": same_game_safe,
            "requires_shift": requires_shift,
            "historical_aggregation_allowed": historical_aggregation_allowed,
            "non_null_rows": int(series.notna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "reason": _classification_reason(
                source_status,
                family,
                column,
            ),
        })

    registry = pd.DataFrame(rows)

    if expected_total_columns is not None and len(registry) != int(expected_total_columns):
        raise AssertionError(
            f"Expected {expected_total_columns} frozen Phase 2D columns, "
            f"found {len(registry)}."
        )

    validate_pregame_identity_source_registry(
        registry,
        expected_source_count=expected_source_count,
    )

    return registry.reset_index(
        drop=True
    )


def approved_lagged_identity_columns(
    registry: pd.DataFrame,
) -> list[str]:
    """Return the approved lagged identity source column names, in the
    registry's (frozen Phase 2D source) row order."""
    required = {"column", "source_status"}
    missing = sorted(
        required.difference(
            registry.columns
        )
    )
    if missing:
        raise KeyError(
            f"Registry missing required columns: {missing}"
        )

    approved = registry.loc[
        registry["source_status"].eq(
            "lagged_identity_source"
        ),
        "column",
    ].tolist()

    return [str(column) for column in approved]


def validate_pregame_identity_source_registry(
    registry: pd.DataFrame,
    expected_source_count: int = EXPECTED_LAGGED_SOURCE_COUNT,
) -> None:
    if registry.empty:
        raise ValueError(
            "Pregame identity source registry is empty."
        )

    if not registry["column"].is_unique:
        raise AssertionError(
            "Registry column names must be unique."
        )

    approved = registry.loc[
        registry["source_status"].eq("lagged_identity_source")
    ]

    if expected_source_count is not None and len(approved) != int(expected_source_count):
        raise AssertionError(
            f"Registry approved-source count mismatch: expected "
            f"{expected_source_count}, got {len(approved)}."
        )

    if approved["same_game_safe"].any():
        raise AssertionError(
            "Approved lagged identity sources cannot be same-game safe "
            "without a shift; same-game facts are not pregame-safe."
        )

    if not approved["requires_shift"].all():
        raise AssertionError(
            "Approved lagged identity sources must all require a "
            "chronological shift."
        )

    identity_keys = registry.loc[
        registry["source_status"].eq("identity_key")
    ]
    if not identity_keys["same_game_safe"].all():
        raise AssertionError(
            "Identity key columns must be marked same-game safe "
            "(they are used only for joining, not as facts)."
        )


def phase_2e_identity_source_registry_paths(
    season: int,
    data_root: Path | None = None,
) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    base = (
        root
        / "game_intelligence"
        / "pregame_identity_registry"
        / str(int(season))
    )

    return {
        "base_dir": base,
        "registry_csv": base / "pregame_identity_source_registry.csv",
        "metadata_json": base / "pregame_identity_source_registry_metadata.json",
    }


def build_pregame_identity_source_registry_metadata(
    registry: pd.DataFrame,
    season: int,
) -> dict[str, object]:
    validate_pregame_identity_source_registry(
        registry,
        expected_source_count=int(
            registry["source_status"].eq("lagged_identity_source").sum()
        ),
    )

    return {
        "phase": "2E.1",
        "season": int(season),
        "total_columns": int(len(registry)),
        "approved_lagged_identity_sources": int(
            registry["source_status"].eq("lagged_identity_source").sum()
        ),
        "identity_key_columns": int(
            registry["source_status"].eq("identity_key").sum()
        ),
        "excluded_columns": int(
            registry["source_status"].eq("exclude").sum()
        ),
        "same_game_identity_sources": int(
            registry.loc[
                registry["source_status"].eq("lagged_identity_source"),
                "same_game_safe",
            ].sum()
        ),
        "future_games_used": False,
        "registry_engine_version": ENGINE_VERSION,
    }


def save_pregame_identity_source_registry(
    registry: pd.DataFrame,
    season: int,
    data_root: Path | None = None,
) -> dict[str, Path]:
    metadata = build_pregame_identity_source_registry_metadata(
        registry,
        season=season,
    )
    paths = phase_2e_identity_source_registry_paths(
        season=season,
        data_root=data_root,
    )

    paths["base_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )
    registry.to_csv(
        paths["registry_csv"],
        index=False,
    )
    paths["metadata_json"].write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return paths
