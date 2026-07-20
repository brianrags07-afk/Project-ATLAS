"""
Historical coverage matrix for 2024/2025/2026, redesigned to emit five
independent evidence dimensions per (row, season) instead of one
collapsed status.

Every dimension is computed from an explicit, documented, and tested
rule. No dimension is derived from another dimension except through the
few explicitly named/tested mapping functions in
``atlas.audit.temporal_proof`` and ``atlas.audit.schedule_source_assessment``.

ATLAS no-leakage rules encoded here:
  - ``published_schedule`` is never marked complete/pregame-safe merely
    because completed games appear in ``master_game_database``.
  - ``published_series_context`` (series length/boundaries) is
    pregame-safe only when sourced from a published schedule or another
    timestamp-proven pregame source; inferred-from-results series
    context is postgame-only and unsafe.
  - Dynamic pregame fields (starters/lineups/bullpen/injuries/weather/
    umpire/rest/travel/market) require per-game timestamp proof before
    they can be called pregame-safe.
  - Final scores / pitch-by-pitch / plate appearances / batted-ball data
    are postgame facts: they may be complete and useful for historical
    reconstruction, but are never pregame-safe.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas.audit.evidence import DIMENSION_VALUES, make_evidence, unknown_dimensions
from atlas.audit.schedule_source_assessment import (
    assess_schedule_source,
    evidence_from_completed_games,
    evidence_from_series_inferred_from_results,
)
from atlas.audit.temporal_proof import (
    assess_field_temporal_availability,
    assess_pregame_safety_from_temporal_availability,
)

SEASONS = (2024, 2025, 2026)

COVERAGE_ROWS = (
    "published_schedule",
    "game_identifiers",
    "scheduled_first_pitch",
    "final_scores",
    "pitch_by_pitch",
    "plate_appearances",
    "batted_ball_data",
    "starters",
    "bullpen_usage",
    "lineups",
    "injuries",
    "weather",
    "venue",
    "umpire",
    "rest",
    "travel",
    "published_series_context",
    "opening_market",
    "closing_market",
    "team_memories",
    "player_memories",
    "identities",
    "concept_discovery",
    "concept_validation",
    "model_artifacts",
    "frozen_predictions",
    "frozen_pregame_cards",
    "total_runs_targets",
    "scoring_shape_classification",
    "projected_team_runs",
    "projected_game_total",
    "over_under_model_readiness",
    "team_total_model_readiness",
)

# Rows whose provenance/pregame-safety must go through the explicit
# schedule-source assessment rather than generic column-presence rules.
SCHEDULE_PROVENANCE_ROWS = ("published_schedule", "game_identifiers", "scheduled_first_pitch")
SERIES_CONTEXT_ROWS = ("published_series_context",)

# Rows that are postgame facts by definition -- never pregame-safe,
# regardless of completeness.
#
# "final_scores" is a game-level fact evidenced by master_game_database's
# final-outcome columns (home_score/away_score/final_score/winning_team/
# result) and is handled by ``_final_scores_row``. The remaining rows are
# pitch/plate-appearance/batted-ball-level facts evidenced by
# master_pitch_database and are handled by ``_pitch_level_fact_row``.
# These two evidence sources must never be conflated: final-score
# availability must not depend on the pitch database, and vice versa.
FINAL_SCORE_ROWS = ("final_scores",)
PITCH_LEVEL_FACT_ROWS = ("pitch_by_pitch", "plate_appearances", "batted_ball_data")
POSTGAME_FACT_ROWS = FINAL_SCORE_ROWS + PITCH_LEVEL_FACT_ROWS

# Rows that are dynamic pregame fields requiring per-game timestamp proof.
DYNAMIC_PREGAME_ROWS = (
    "starters",
    "bullpen_usage",
    "lineups",
    "injuries",
    "weather",
    "umpire",
    "rest",
    "travel",
    "opening_market",
    "closing_market",
)

# Rows evidenced by repository modules only (no season-specific dataset
# evidence available to this audit).
MODULE_ONLY_ROWS = (
    "team_memories",
    "player_memories",
    "identities",
    "concept_discovery",
    "concept_validation",
    "model_artifacts",
    "frozen_predictions",
    "frozen_pregame_cards",
    "total_runs_targets",
    "scoring_shape_classification",
    "projected_team_runs",
    "projected_game_total",
    "over_under_model_readiness",
    "team_total_model_readiness",
)

ROW_TO_FEATURE_PRESENCE_KEY = {
    "final_scores": "final_outcomes",
    "starters": "starter_information",
    "bullpen_usage": "bullpen_usage",
    "lineups": "lineups",
    "injuries": "injuries",
    "weather": "weather",
    "venue": "venue",
    "umpire": "umpire",
    "rest": "rest",
    "travel": "travel",
    "opening_market": "market_data",
    "closing_market": "market_data",
}

ROW_TO_FOCUS_AREA = {
    "team_memories": "memories",
    "player_memories": "memories",
    "identities": "identities",
    "concept_discovery": "concepts",
    "concept_validation": "validation",
    "model_artifacts": "prediction",
    "frozen_predictions": "prediction",
    "frozen_pregame_cards": "pregame_snapshots",
    # Totals/scoring-shape is a first-class target family, independent of
    # moneyline/run-margin (see atlas/learning/totals_target_builder.py).
    # All six totals-readiness rows map to the "totals" focus area so this
    # audit reports evidence separately from generic "prediction" rows.
    "total_runs_targets": "totals",
    "scoring_shape_classification": "totals",
    "projected_team_runs": "totals",
    "projected_game_total": "totals",
    "over_under_model_readiness": "totals",
    "team_total_model_readiness": "totals",
}

DIMENSION_KEYS = (
    "data_presence",
    "source_completeness",
    "provenance_status",
    "temporal_availability",
    "pregame_safety",
)


def _season_has_rows(rows_by_season: dict[str, int], season: int) -> bool:
    return rows_by_season.get(str(season), 0) > 0


def _unique_games_for_season(profile: dict[str, Any] | None, season: int) -> int | None:
    """Return the observed unique-``game_pk`` count for ``season`` from a
    dataset profile, or ``None`` when no such evidence exists. Prefer this
    over raw row counts for any game-level completeness comparison -- a
    dataset can have many rows per game (e.g. one row per team, or one row
    per pitch) without that implying anything about season coverage."""
    if not profile:
        return None
    counts = profile.get("unique_games_by_season") or {}
    if str(season) not in counts:
        return None
    return int(counts[str(season)])


def _source_completeness_from_game_counts(
    observed_games: int | None, expected_games: int | None
) -> str:
    """Compare observed unique-game coverage against an explicit expected
    game count for a season. ``complete`` is returned only when both an
    expected count is known and the observed count meets or exceeds it --
    a single observed row or a single represented game can never produce
    ``complete`` when more than one game is expected. When no reliable
    expected-count reference is available, but data exists, the result is
    ``unknown`` rather than an assumed ``complete``."""
    if not observed_games or observed_games <= 0:
        return "not_applicable"
    if expected_games is None or expected_games <= 0:
        return "unknown"
    if observed_games >= expected_games:
        return "complete"
    return "partial"


def _find_dataset_evidence(
    row: str,
    season: int,
    dataset_profiles: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    """Return (evidence_records, data_presence, source_completeness) for a
    generic column-mapped row."""
    feature_key = ROW_TO_FEATURE_PRESENCE_KEY.get(row)
    evidence: list[dict[str, Any]] = []
    if feature_key is None:
        return evidence, "unknown", "unknown"

    matches = []
    for dataset_name, profile in dataset_profiles.items():
        column = profile.get("feature_presence", {}).get(feature_key)
        has_season_rows = _season_has_rows(profile.get("rows_by_season", {}), season)
        if column and has_season_rows:
            matches.append((dataset_name, column, profile))

    if not matches:
        for dataset_name, profile in dataset_profiles.items():
            column = profile.get("feature_presence", {}).get(feature_key)
            if column:
                evidence.append(
                    make_evidence(
                        "column_presence",
                        source=dataset_name,
                        field_or_column=column,
                        season=season,
                        confidence="observed",
                        limitation=f"column exists but no rows observed for season {season}",
                    )
                )
                return evidence, "unknown", "unknown"
        return evidence, "missing", "not_applicable"

    dataset_name, column, profile = matches[0]
    row_count = profile.get("rows_by_season", {}).get(str(season), 0)
    null_pct = profile.get("null_percentages", {}).get(column)
    evidence.append(
        make_evidence(
            "column_presence",
            source=dataset_name,
            field_or_column=column,
            season=season,
            observed_value={"row_count": row_count, "null_percentage": null_pct},
            confidence="observed",
        )
    )
    data_presence = "present"
    if null_pct is None:
        source_completeness = "unknown"
    elif null_pct == 0:
        source_completeness = "complete"
    elif null_pct >= 90:
        source_completeness = "incomplete"
    else:
        source_completeness = "partial"
    return evidence, data_presence, source_completeness


def _classification_for(dataset_profiles: dict[str, dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    for record in evidence:
        dataset_name = record.get("source")
        column = record.get("field_or_column")
        profile = dataset_profiles.get(dataset_name, {})
        classification = profile.get("column_classification", {}).get(column)
        if classification:
            return classification
    return "unknown"


def _module_row(row: str, season: int, repository_inventory: dict[str, Any]) -> dict[str, Any]:
    focus_area = ROW_TO_FOCUS_AREA.get(row)
    modules = repository_inventory.get("focus_area_index", {}).get(focus_area, []) if focus_area else []
    dims = unknown_dimensions()
    if not modules:
        evidence: list[dict[str, Any]] = []
        dims["data_presence"] = "missing"
        dims["source_completeness"] = "not_applicable"
        risks = [f"No repository module implements focus area '{focus_area}'."]
        required = [f"Implement or locate a module producing '{row}' for season {season}."]
    else:
        evidence = [
            make_evidence(
                "repository_module",
                source=", ".join(modules),
                season=season,
                observed_value={"modules": modules},
                confidence="heuristic",
                limitation=(
                    "Module existence in the repository does not prove season-specific "
                    "artifact production; this audit inspects source code only, not "
                    "produced artifacts."
                ),
            )
        ]
        dims["data_presence"] = "unknown"
        dims["source_completeness"] = "unknown"
        risks = [
            f"Module(s) {modules} exist for focus area '{focus_area}', but this audit did "
            f"not directly observe a produced, season-{season} artifact."
        ]
        required = [
            f"Locate or produce a manifest-linked artifact for '{row}' season {season} "
            "and re-run this audit against it."
        ]
    dims["provenance_status"] = "missing" if not modules else "unknown"
    if row == "frozen_pregame_cards":
        dims["temporal_availability"] = "unknown"
        dims["pregame_safety"] = "unknown"
    else:
        dims["temporal_availability"] = "not_applicable"
        dims["pregame_safety"] = "not_applicable"
    return {
        "row": row,
        "season": season,
        **dims,
        "evidence": evidence,
        "risks": risks,
        "required_next_evidence": required,
    }


def _expected_games_for_season(dataset_profiles: dict[str, dict[str, Any]], season: int) -> int | None:
    """Return an independently-sourced expected-game-count for ``season``,
    or ``None`` when no such reference is available.

    This must come from a source distinct from ``master_game_database``'s
    own observed ``unique_games_by_season`` -- using that same value as
    both the observed and the expected count would be circular (it would
    prove only that the games present are internally self-consistent, not
    that the season's full game universe is represented). A genuine
    expected-count reference (e.g. a published season schedule / league
    game-count registry) is expected under the ``season_schedule`` key."""
    schedule_profile = dataset_profiles.get("season_schedule")
    if not schedule_profile:
        return None
    counts = schedule_profile.get("expected_games_by_season") or {}
    if str(season) not in counts:
        return None
    return int(counts[str(season)])


def _null_completeness_from_percentage(null_pct: float | None) -> str:
    """Evaluate final-outcome column null-percentage in isolation from
    game-count coverage. This proves only that the score field is
    populated for whichever rows are present -- it says nothing about
    whether every game in the season is represented."""
    if null_pct is None:
        return "unknown"
    if null_pct >= 100:
        return "not_applicable"
    if null_pct == 0:
        return "complete"
    return "partial"


def _final_scores_row(season: int, dataset_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evidence for the ``final_scores`` row comes exclusively from
    ``master_game_database``'s game-level final-outcome columns
    (home_score/away_score/final_score/winning_team/result via the
    ``final_outcomes`` feature-presence key). This must never depend on
    ``master_pitch_database`` -- pitch-level presence is neither necessary
    nor sufficient evidence that final-score outcomes are present."""
    game_profile = dataset_profiles.get("master_game_database")
    score_column = None
    if game_profile:
        score_column = game_profile.get("feature_presence", {}).get("final_outcomes")

    dims = unknown_dimensions()
    has_rows = bool(game_profile) and _season_has_rows(game_profile.get("rows_by_season", {}), season)

    if score_column and has_rows:
        row_count = game_profile.get("rows_by_season", {}).get(str(season), 0)
        null_pct = game_profile.get("null_percentages", {}).get(score_column)
        observed_games = _unique_games_for_season(game_profile, season)
        expected_games = _expected_games_for_season(dataset_profiles, season)
        evidence = [
            make_evidence(
                "column_presence",
                source="master_game_database",
                field_or_column=score_column,
                season=season,
                observed_value={
                    "row_count": row_count,
                    "null_percentage": null_pct,
                    "unique_games_observed": observed_games,
                    "unique_games_expected": expected_games,
                },
                confidence="observed",
                limitation="Postgame final-outcome fact; never a same-game pregame input.",
            )
        ]
        dims["data_presence"] = "present"
        # Null completeness (is the populated data fully non-null?) and
        # game coverage (do the observed games match an independent
        # expected-game-count reference?) are evaluated as two separate
        # checks. A "0% null" reading proves only that the score field is
        # populated for the games that happen to be present in
        # master_game_database -- it is never, by itself, evidence that
        # every game in the season is represented. "complete" requires
        # both checks to pass; a null-percentage of 0 with no independent
        # expected-count reference stays "unknown" rather than assumed
        # complete.
        if observed_games is None:
            dims["source_completeness"] = "unknown"
        else:
            null_completeness = _null_completeness_from_percentage(null_pct)
            if null_completeness in ("unknown", "not_applicable"):
                dims["source_completeness"] = null_completeness
            elif null_completeness == "partial":
                dims["source_completeness"] = "partial"
            else:
                dims["source_completeness"] = _source_completeness_from_game_counts(
                    observed_games, expected_games
                )
        dims["provenance_status"] = "partial"
        risks = [
            f"'final_scores' for season {season} is complete/present but is a postgame fact. "
            "It may support historical reconstruction/learning but must never serve as a "
            "same-game pregame input."
        ]
        required = [
            "No further evidence required to use this as a postgame reconstruction input; "
            "pregame use is never authorized regardless of additional evidence."
        ]
    else:
        evidence = []
        dims["data_presence"] = "missing"
        dims["source_completeness"] = "not_applicable"
        dims["provenance_status"] = "missing"
        risks = [f"No master_game_database final-outcome columns/rows found for season {season}."]
        required = [
            f"Locate final-outcome fields (home_score/away_score/final_score/winning_team/"
            f"result) in master_game_database for season {season}."
        ]
    dims["temporal_availability"] = "postgame_only"
    dims["pregame_safety"] = "unsafe" if dims["data_presence"] != "missing" else "not_applicable"
    return {
        "row": "final_scores",
        "season": season,
        **dims,
        "evidence": evidence,
        "risks": risks,
        "required_next_evidence": required,
    }


def _pitch_level_fact_row(
    row: str, season: int, dataset_profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Evidence for pitch_by_pitch/plate_appearances/batted_ball_data comes
    from ``master_pitch_database``. Row presence alone never proves
    complete season coverage: the observed unique-``game_pk`` count for
    ``master_pitch_database`` is compared against the expected game count
    from ``master_game_database`` (the season's game universe) where that
    reference is available; otherwise completeness stays ``unknown``."""
    pitch_profile = dataset_profiles.get("master_pitch_database")
    game_profile = dataset_profiles.get("master_game_database")
    pitch_season_counts: dict[str, int] = {}
    if pitch_profile:
        pitch_season_counts = pitch_profile.get("pitches_by_season") or pitch_profile.get("rows_by_season", {})
    dims = unknown_dimensions()
    if _season_has_rows(pitch_season_counts, season):
        row_count = pitch_season_counts.get(str(season), 0)
        observed_games = _unique_games_for_season(pitch_profile, season)
        expected_games = _unique_games_for_season(game_profile, season)
        evidence = [
            make_evidence(
                "row_presence",
                source="master_pitch_database",
                season=season,
                observed_value={
                    "row_count": row_count,
                    "unique_games_observed": observed_games,
                    "unique_games_expected": expected_games,
                },
                confidence="observed",
                limitation=(
                    "Postgame pitch-level fact; never a same-game pregame input. Row "
                    "presence alone does not prove full-season game coverage."
                ),
            )
        ]
        dims["data_presence"] = "present"
        dims["source_completeness"] = _source_completeness_from_game_counts(observed_games, expected_games)
        dims["provenance_status"] = "partial"
        risks = [
            f"'{row}' for season {season} is complete/present but is a postgame fact. "
            "It may support historical reconstruction/learning but must never serve as a "
            "same-game pregame input."
        ]
        required = [
            "No further evidence required to use this as a postgame reconstruction input; "
            "pregame use is never authorized regardless of additional evidence."
        ]
        if dims["source_completeness"] == "unknown":
            required.append(
                f"Supply a reliable expected-game-count reference (e.g. master_game_database "
                f"unique game counts) for season {season} to evaluate '{row}' coverage."
            )
    else:
        evidence = []
        dims["data_presence"] = "missing"
        dims["source_completeness"] = "not_applicable"
        dims["provenance_status"] = "missing"
        risks = [f"No master_pitch_database rows found for season {season}."]
        required = [f"Locate raw/master pitch-level source data for season {season}."]
    dims["temporal_availability"] = "postgame_only"
    dims["pregame_safety"] = "unsafe" if dims["data_presence"] != "missing" else "not_applicable"
    return {
        "row": row,
        "season": season,
        **dims,
        "evidence": evidence,
        "risks": risks,
        "required_next_evidence": required,
    }



def _schedule_provenance_row(
    row: str, season: int, dataset_profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    schedule_profile = dataset_profiles.get("master_game_database")
    evidence: list[dict[str, Any]] = []
    has_rows = bool(schedule_profile) and _season_has_rows(schedule_profile.get("rows_by_season", {}), season)
    dims = unknown_dimensions()

    if has_rows:
        row_count = schedule_profile.get("rows_by_season", {}).get(str(season), 0)
        evidence.append(evidence_from_completed_games("master_game_database", season, row_count))
        dims["data_presence"] = "present"
        # Row presence alone is not proof of complete season coverage --
        # this audit has no independent, explicit expected-game-count
        # reference for the schedule itself (master_game_database IS the
        # candidate schedule source being assessed), so completeness stays
        # "unknown" rather than being inferred from row_count > 0.
        dims["source_completeness"] = "unknown"
    else:
        dims["data_presence"] = "missing"
        dims["source_completeness"] = "not_applicable"

    assessment = assess_schedule_source(evidence)
    dims["provenance_status"] = assessment["provenance_status"]
    dims["temporal_availability"] = assessment["temporal_availability"]
    dims["pregame_safety"] = assessment["pregame_safety"]

    risks = []
    required = []
    if has_rows:
        risks.append(
            f"'{row}' for season {season}: completed games observed in master_game_database, "
            "but this is NOT proof of a published, pregame schedule. "
            "provenance_status/pregame_safety remain unverified/unsafe unless a timestamped "
            "published-schedule source is supplied."
        )
        required.append(
            "Supply a timestamped published-schedule source (source_retrieved_at_utc "
            "on-or-before each game's scheduled start) to verify provenance and pregame safety."
        )
    else:
        risks.append(f"No master_game_database rows found for season {season}.")
        required.append(f"Locate master_game_database (or a raw schedule source) rows for season {season}.")

    return {
        "row": row,
        "season": season,
        **dims,
        "evidence": evidence,
        "risks": risks,
        "required_next_evidence": required,
    }


def _series_context_row(season: int, dataset_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    schedule_profile = dataset_profiles.get("master_game_database")
    series_column = None
    if schedule_profile:
        series_column = schedule_profile.get("feature_presence", {}).get("published_series_context")

    dims = unknown_dimensions()
    evidence: list[dict[str, Any]] = []

    if series_column:
        has_rows = _season_has_rows(schedule_profile.get("rows_by_season", {}), season)
        if has_rows:
            dims["data_presence"] = "present"
            # No independent expected-count reference exists for series
            # length/boundaries in this audit -- row presence alone must
            # never be treated as "complete" season coverage.
            dims["source_completeness"] = "unknown"
            # This audit cannot verify whether the column came from a
            # timestamped published schedule vs. being back-filled from
            # completed results -- treat conservatively as inferred from
            # results unless explicit published-schedule-source evidence
            # is supplied elsewhere.
            evidence.append(evidence_from_series_inferred_from_results("master_game_database", season))
        else:
            dims["data_presence"] = "missing"
            dims["source_completeness"] = "not_applicable"
    else:
        dims["data_presence"] = "missing"
        dims["source_completeness"] = "not_applicable"

    assessment = assess_schedule_source(evidence)
    dims["provenance_status"] = assessment["provenance_status"]
    dims["temporal_availability"] = assessment["temporal_availability"]
    dims["pregame_safety"] = assessment["pregame_safety"]

    risks = [
        "Series length/boundaries inferred from completed game history are postgame-only "
        "and must never authorize pregame prediction of series context.",
    ]
    required = [
        "Supply a timestamped published-schedule (or other pregame) series-context source "
        "to verify provenance and pregame safety; do not infer series length from results.",
    ]
    return {
        "row": "published_series_context",
        "season": season,
        **dims,
        "evidence": evidence,
        "risks": risks,
        "required_next_evidence": required,
    }


def _generic_dataset_row(
    row: str, season: int, dataset_profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    evidence, data_presence, source_completeness = _find_dataset_evidence(row, season, dataset_profiles)
    classification = _classification_for(dataset_profiles, evidence)
    dims = unknown_dimensions()
    dims["data_presence"] = data_presence
    dims["source_completeness"] = source_completeness
    dims["provenance_status"] = "partial" if data_presence == "present" else (
        "missing" if data_presence == "missing" else "unknown"
    )

    is_dynamic = row in DYNAMIC_PREGAME_ROWS
    if data_presence != "present":
        temporal_availability, _reason = "unknown", "no evidence"
    elif classification == "postgame_fact":
        temporal_availability, _reason = "postgame_only", "column classified as a postgame fact"
    elif classification == "identifier":
        temporal_availability, _reason = "not_applicable", "identifier column, not a dynamic pregame fact"
    else:
        # pregame_possible_but_needs_timestamp_proof / schedule_safe / unknown:
        # no per-game source_retrieved_at evidence has been supplied to this
        # audit for these dynamic fields, so temporal availability stays
        # unknown rather than being assumed safe.
        temporal_availability, _reason = assess_field_temporal_availability(evidence)

    dims["temporal_availability"] = temporal_availability
    dims["pregame_safety"] = assess_pregame_safety_from_temporal_availability(
        temporal_availability, is_dynamic_pregame_field=is_dynamic
    )

    risks = []
    required = []
    if data_presence == "missing":
        risks.append(f"No dataset column found for '{row}' in season {season}.")
        required.append(f"Locate a raw or master source containing '{row}' for season {season}.")
    elif is_dynamic and dims["pregame_safety"] in ("unknown", "conditional"):
        risks.append(
            f"'{row}' is present for season {season} but has no per-game "
            "source_retrieved_at_utc timestamp proving it was known before "
            "feature_cutoff_time; it is not pregame-safe until proven."
        )
        required.append(
            f"Attach field-level source_retrieved_at_utc timestamps for '{row}' season {season} "
            "and verify they are on-or-before each game's feature_cutoff_time."
        )
    elif dims["pregame_safety"] == "unsafe":
        risks.append(f"'{row}' for season {season} is a postgame fact and is never pregame-safe.")

    return {
        "row": row,
        "season": season,
        **dims,
        "evidence": evidence,
        "risks": risks,
        "required_next_evidence": required,
    }


def build_coverage_matrix(
    dataset_profiles: dict[str, dict[str, Any]],
    repository_inventory: dict[str, Any],
    cloud_inventory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_name in COVERAGE_ROWS:
        for season in SEASONS:
            if row_name in SCHEDULE_PROVENANCE_ROWS:
                entry = _schedule_provenance_row(row_name, season, dataset_profiles)
            elif row_name in SERIES_CONTEXT_ROWS:
                entry = _series_context_row(season, dataset_profiles)
            elif row_name in FINAL_SCORE_ROWS:
                entry = _final_scores_row(season, dataset_profiles)
            elif row_name in PITCH_LEVEL_FACT_ROWS:
                entry = _pitch_level_fact_row(row_name, season, dataset_profiles)
            elif row_name in MODULE_ONLY_ROWS:
                entry = _module_row(row_name, season, repository_inventory)
            else:
                entry = _generic_dataset_row(row_name, season, dataset_profiles)

            for key in DIMENSION_KEYS:
                if entry[key] not in DIMENSION_VALUES[key]:
                    entry[key] = "unknown" if "unknown" in DIMENSION_VALUES[key] else DIMENSION_VALUES[key][0]
            rows.append(entry)
    return rows


def write_coverage_matrix(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["row", "season", *DIMENSION_KEYS, "risks", "required_next_evidence"]
    csv_path = output_dir / "historical_coverage_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "row": r["row"],
                "season": r["season"],
                **{k: r[k] for k in DIMENSION_KEYS},
                "risks": " | ".join(r.get("risks", [])),
                "required_next_evidence": " | ".join(r.get("required_next_evidence", [])),
            })

    json_path = output_dir / "historical_coverage_matrix.json"
    json_path.write_text(
        json.dumps(
            {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# ATLAS Historical Coverage Matrix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Each row/season carries five independent dimensions: data_presence, "
        "source_completeness, provenance_status, temporal_availability, pregame_safety. "
        "See `historical_coverage_matrix.json` for full evidence records.",
        "",
        "| Row | Season | Data presence | Completeness | Provenance | Temporal | Pregame safety |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['row']} | {r['season']} | {r['data_presence']} | {r['source_completeness']} | "
            f"{r['provenance_status']} | {r['temporal_availability']} | {r['pregame_safety']} |"
        )
    md_path = output_dir / "historical_coverage_matrix.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return csv_path, md_path
