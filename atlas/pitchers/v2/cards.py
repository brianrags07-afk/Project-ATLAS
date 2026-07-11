
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from atlas.config import DATA_DIR
from atlas.pitchers.v2.definitions import (
    PITCHER_CARD_VERSION,
    PITCHER_ENGINE_VERSION,
)
from atlas.pitchers.v2.pitch_table import MASTER_PITCH_PATH
from atlas.pitchers.v2.summaries import (
    grouped_summaries,
    nested_summaries,
    summarize_pitcher_evidence,
)


GAME_CARD_MANIFEST = (
    DATA_DIR
    / "history"
    / "game_cards"
    / "game_card_manifest.parquet"
)


def build_appearance_timeline(
    pitcher_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    appearances = []

    grouped = []

    for game_pk, game_df in pitcher_df.groupby(
        "game_pk",
        sort=False,
    ):
        game_df = game_df.sort_values(
            [
                "inning",
                "at_bat_number",
                "pitch_number",
            ],
            kind="stable",
        )

        first = game_df.iloc[0]
        game_date = pd.Timestamp(first["game_date"])

        grouped.append({
            "game_pk": int(game_pk),
            "game_date": game_date,
            "game_df": game_df,
            "first": first,
        })

    grouped.sort(
        key=lambda item: (
            item["game_date"],
            item["game_pk"],
        )
    )

    prior_appearances = []

    for item in grouped:
        game_pk = item["game_pk"]
        game_date = item["game_date"]
        game_df = item["game_df"]
        first = item["first"]

        previous = (
            prior_appearances[-1]
            if prior_appearances
            else None
        )

        rest_days = None

        if previous is not None:
            rest_days = int(
                (
                    game_date.normalize()
                    - previous["game_date"].normalize()
                ).days
            )

        def workload_in_days(days: int) -> dict[str, int]:
            cutoff = game_date.normalize() - pd.Timedelta(
                days=days
            )

            eligible = [
                appearance
                for appearance in prior_appearances
                if (
                    appearance["game_date"].normalize()
                    >= cutoff
                    and appearance["game_date"].normalize()
                    < game_date.normalize()
                )
            ]

            return {
                "appearances": int(len(eligible)),
                "pitches": int(
                    sum(
                        appearance["pitches"]
                        for appearance in eligible
                    )
                ),
            }

        summary = summarize_pitcher_evidence(
            game_df
        )

        innings = pd.to_numeric(
            game_df["inning"],
            errors="coerce",
        )

        at_bats = game_df[
            [
                "game_pk",
                "pitcher_id",
                "at_bat_number",
            ]
        ].drop_duplicates()

        appearance = {
            "game_pk": game_pk,
            "date": (
                game_date.date().isoformat()
                if pd.notna(game_date)
                else None
            ),
            "season": int(
                first["atlas_season"]
            ),
            "team": first.get("team"),
            "opponent": first.get("opponent"),
            "home_away": first.get("home_away"),
            "role": first.get("role"),
            "throws": first.get("throws"),

            "first_inning": (
                int(innings.min())
                if innings.notna().any()
                else None
            ),
            "last_inning": (
                int(innings.max())
                if innings.notna().any()
                else None
            ),
            "pitches": int(len(game_df)),
            "batters_faced": int(len(at_bats)),

            "rest": {
                "days_since_previous_appearance": rest_days,
                "back_to_back": (
                    rest_days == 1
                    if rest_days is not None
                    else False
                ),
                "short_rest": (
                    rest_days is not None
                    and rest_days <= 3
                ),
                "long_rest": (
                    rest_days is not None
                    and rest_days >= 7
                ),
                "previous_3_days": workload_in_days(3),
                "previous_7_days": workload_in_days(7),
                "previous_14_days": workload_in_days(14),
            },

            "summary": summary,

            "pitch_count_progression": {
                "max_exact_pitch_count": int(
                    game_df[
                        "appearance_pitch_count"
                    ].max()
                ),
                "buckets_reached": [
                    str(value)
                    for value in game_df[
                        "pitch_count_bucket"
                    ].dropna().unique()
                ],
            },

            "game_card_reference": {
                "game_pk": game_pk,
                "manifest": str(
                    GAME_CARD_MANIFEST
                ),
                "season_event_store": int(
                    first["atlas_season"]
                ),
                "lookup_key": str(game_pk),
            },
        }

        appearances.append(appearance)

        prior_appearances.append({
            "game_date": game_date,
            "pitches": int(len(game_df)),
        })

    return appearances


def build_pitcher_card(
    pitcher_df: pd.DataFrame,
) -> dict[str, Any]:
    if pitcher_df.empty:
        raise ValueError("pitcher_df is empty")

    pitcher_df = pitcher_df.sort_values(
        [
            "game_date",
            "game_pk",
            "inning",
            "at_bat_number",
            "pitch_number",
        ],
        kind="stable",
    ).copy()

    first = pitcher_df.iloc[0]
    last = pitcher_df.iloc[-1]

    pitcher_id = int(first["pitcher_id"])

    seasons = sorted(
        int(value)
        for value in pitcher_df[
            "atlas_season"
        ].dropna().unique()
    )

    teams = sorted(
        str(value)
        for value in pitcher_df[
            "team"
        ].dropna().unique()
    )

    timeline = build_appearance_timeline(
        pitcher_df
    )

    card = {
        "metadata": {
            "pitcher_card_version": PITCHER_CARD_VERSION,
            "pitcher_engine_version": PITCHER_ENGINE_VERSION,
            "pitcher_id": pitcher_id,
            "pitcher_name": None,
            "throws": first.get("throws"),
            "created_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "updated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "regular_season_only": True,
            "game_type_filter": "R",
        },

        "scope": {
            "seasons": seasons,
            "teams": teams,
            "date_start": (
                pitcher_df["game_date"]
                .min()
                .date()
                .isoformat()
            ),
            "date_end": (
                pitcher_df["game_date"]
                .max()
                .date()
                .isoformat()
            ),
            "current_team": last.get("team"),
            "current_role": last.get("role"),
        },

        "overall": summarize_pitcher_evidence(
            pitcher_df
        ),

        "historical_splits": {
            "season": grouped_summaries(
                pitcher_df,
                "atlas_season",
            ),
            "team": grouped_summaries(
                pitcher_df,
                "team",
            ),
            "role": grouped_summaries(
                pitcher_df,
                "role",
            ),
            "home_away": grouped_summaries(
                pitcher_df,
                "home_away",
            ),
            "batter_handedness": grouped_summaries(
                pitcher_df,
                "batter_side",
            ),
            "inning": grouped_summaries(
                pitcher_df,
                "inning",
            ),
            "base_state": grouped_summaries(
                pitcher_df,
                "base_state",
            ),
            "score_state": grouped_summaries(
                pitcher_df,
                "score_state",
            ),
        },

        "arsenal": {
            "by_pitch_type": grouped_summaries(
                pitcher_df,
                "pitch_type",
            ),
            "pitch_type_by_handedness": nested_summaries(
                pitcher_df,
                "pitch_type",
                "batter_side",
            ),
            "pitch_type_by_count_state": nested_summaries(
                pitcher_df,
                "pitch_type",
                "count_state",
            ),
            "pitch_type_by_exact_count": nested_summaries(
                pitcher_df,
                "pitch_type",
                "exact_count",
            ),
            "pitch_type_by_pitch_count": nested_summaries(
                pitcher_df,
                "pitch_type",
                "pitch_count_bucket",
            ),
            "pitch_type_by_times_through_order": nested_summaries(
                pitcher_df,
                "pitch_type",
                "times_through_order",
            ),
            "pitch_type_by_role": nested_summaries(
                pitcher_df,
                "pitch_type",
                "role",
            ),
        },

        "count_and_execution": {
            "count_state": grouped_summaries(
                pitcher_df,
                "count_state",
            ),
            "exact_count": grouped_summaries(
                pitcher_df,
                "exact_count",
            ),
            "pitch_count_bucket": grouped_summaries(
                pitcher_df,
                "pitch_count_bucket",
            ),
            "times_through_order": grouped_summaries(
                pitcher_df,
                "times_through_order",
            ),
            "pitch_count_by_count_state": nested_summaries(
                pitcher_df,
                "pitch_count_bucket",
                "count_state",
            ),
            "pitch_count_by_handedness": nested_summaries(
                pitcher_df,
                "pitch_count_bucket",
                "batter_side",
            ),
            "pitch_count_by_times_through_order": nested_summaries(
                pitcher_df,
                "pitch_count_bucket",
                "times_through_order",
            ),
            "pitch_count_by_inning": nested_summaries(
                pitcher_df,
                "pitch_count_bucket",
                "inning",
            ),
            "pitch_count_by_base_state": nested_summaries(
                pitcher_df,
                "pitch_count_bucket",
                "base_state",
            ),
            "pitch_count_by_score_state": nested_summaries(
                pitcher_df,
                "pitch_count_bucket",
                "score_state",
            ),
        },

        "timeline": timeline,

        "current_state": {
            "latest_appearance": (
                timeline[-1]
                if timeline
                else None
            ),
            "identity_version": None,
            "transition_confirmed": False,
            "availability_grade": None,
            "pregame_grade": None,
        },

        "situational_evidence_status": {
            "all_splits_are_candidate_evidence": True,
            "predictive_importance_assumed": False,
            "single_metric_control_allowed": False,
            "cross_pitcher_transfer_assumed": False,
            "cross_team_transfer_assumed": False,
            "requires_pitcher_local_validation": True,
            "requires_target_specific_validation": True,
        },

        "unknown_evidence": {
            "status": "preserved_for_future_testing",
            "raw_pitch_events_available": True,
            "deeper_intersections_materialized_on_demand": True,
            "relationships_validated": False,
        },

        "evidence_links": [],
        "validation_links": [],
        "transition_links": [],

        "traceability": {
            "source_dataset": str(
                MASTER_PITCH_PATH
            ),
            "game_card_manifest": str(
                GAME_CARD_MANIFEST
            ),
            "source_event_rows": int(
                len(pitcher_df)
            ),
            "source_game_count": int(
                pitcher_df[
                    "game_pk"
                ].nunique()
            ),
            "source_game_pks": sorted(
                int(value)
                for value in pitcher_df[
                    "game_pk"
                ].dropna().unique()
            ),
            "full_pitch_events_stored_in_game_cards": True,
            "pitch_rows_embedded_in_card": False,
            "exact_pitch_count_preserved_in_source": True,
            "pitch_event_keys_preserved_in_source": True,
        },

        "provenance": {
            "source": (
                "clean regular-season "
                "master_pitch_database.parquet"
            ),
            "built_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        },
    }

    return card
