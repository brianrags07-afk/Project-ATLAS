"""
Validity tests for schemas/pregame_game_card.schema.json.

Uses only the standard library (no jsonschema dependency), consistent
with the existing scripts/dev_data_bundle/manifest.py convention of
validating with plain Python rather than adding a new dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "pregame_game_card.schema.json"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _valid_card() -> dict:
    return {
        "card_identity": {
            "card_id": "card_2025_000001_v1",
            "game_pk": 700123,
            "season": 2025,
            "game_date": "2025-04-01",
            "scheduled_start_time_utc": "2025-04-01T18:05:00Z",
            "home_team": "NYY",
            "away_team": "BOS",
            "venue": "Yankee Stadium",
            "card_version": 1,
        },
        "temporal_provenance": {
            "card_created_at_utc": "2025-04-01T12:00:00Z",
            "feature_cutoff_time_utc": "2025-04-01T17:00:00Z",
            "source_retrieved_at_utc": {"starters": "2025-04-01T16:00:00Z"},
            "code_commit_sha": "abc1234",
            "pipeline_version": "1.0.0",
            "source_manifest_id": "manifest_2025_000001",
            "pregame_safe": True,
            "leakage_audit_status": "passed",
        },
        "schedule_context": {
            "published_series_length": 3,
            "series_game_number": 1,
            "home_rest_days": 1,
            "away_rest_days": 1,
            "home_travel_context": "home_stand",
            "away_travel_context": "traveling",
            "doubleheader_status": "none",
            "schedule_source": "published_schedule_v1",
            "schedule_source_timestamp_utc": "2025-03-01T00:00:00Z",
        },
        "starters": {
            "home": {
                "status": "confirmed",
                "confirmation_status": "confirmed",
                "source_timestamp_utc": "2025-04-01T16:00:00Z",
                "handedness": "R",
                "uncertainty_flags": [],
            },
            "away": {
                "status": "expected",
                "confirmation_status": "probable",
                "source_timestamp_utc": "2025-04-01T14:00:00Z",
                "handedness": "L",
                "uncertainty_flags": ["unconfirmed_rotation"],
            },
        },
        "lineups": {
            "home": {
                "status": "expected",
                "source_timestamp_utc": "2025-04-01T15:00:00Z",
                "batting_order": [],
                "completeness": "partial",
                "uncertainty_flags": [],
            },
            "away": {
                "status": "expected",
                "source_timestamp_utc": "2025-04-01T15:00:00Z",
                "batting_order": [],
                "completeness": "missing",
                "uncertainty_flags": ["no_lineup_posted_yet"],
            },
        },
        "bullpen": {
            "home": {
                "prior_usage_only": True,
                "pitch_counts": {},
                "appearances": {},
                "uncertainty_flags": [],
            },
            "away": {
                "prior_usage_only": True,
                "pitch_counts": {},
                "appearances": {},
                "uncertainty_flags": [],
            },
        },
        "team_and_player_memories": {
            "team_memories": {
                "values": {},
                "observation_count": 10,
                "recency": "2025-03-31T00:00:00Z",
                "sample_sufficiency": "sufficient",
                "version": "1.0.0",
            },
            "player_memories": {
                "values": {},
                "observation_count": 5,
                "recency": "2025-03-31T00:00:00Z",
                "sample_sufficiency": "unknown",
                "version": "1.0.0",
            },
        },
        "environment": {
            "venue": "Yankee Stadium",
            "park_factors": None,
            "weather": None,
            "roof_status": "not_applicable",
            "umpire": None,
            "source_timestamps_utc": {},
            "missingness_flags": ["weather_not_yet_available"],
        },
        "predictions": {
            "model_versions": {"win_probability_model": "1.2.0"},
        },
        "postgame": None,
    }


def test_schema_file_exists_and_is_valid_json():
    assert SCHEMA_PATH.exists()
    schema = _load_schema()
    assert schema["$schema"].startswith("https://json-schema.org/")


def test_schema_declares_required_top_level_sections():
    schema = _load_schema()
    required = set(schema["required"])
    expected = {
        "card_identity",
        "temporal_provenance",
        "schedule_context",
        "starters",
        "lineups",
        "bullpen",
        "team_and_player_memories",
        "environment",
        "predictions",
        "postgame",
    }
    assert expected.issubset(required)


def test_schema_market_is_optional_and_isolated_by_default():
    schema = _load_schema()
    assert "market" not in schema["required"]
    market_props = schema["properties"]["market"]["properties"]
    assert market_props["included_in_baseball_model"]["const"] is False


def test_schema_bullpen_requires_prior_usage_only_true():
    schema = _load_schema()
    bullpen_entry = schema["$defs"]["bullpen_entry"]
    assert bullpen_entry["properties"]["prior_usage_only"]["const"] is True
    assert "prior_usage_only" in bullpen_entry["required"]


def test_schema_postgame_is_always_null():
    schema = _load_schema()
    assert schema["properties"]["postgame"]["const"] is None


def test_valid_card_fixture_has_all_required_top_level_keys():
    schema = _load_schema()
    card = _valid_card()
    for key in schema["required"]:
        assert key in card, f"missing required top-level key: {key}"


def test_valid_card_fixture_starters_have_no_extra_keys():
    schema = _load_schema()
    allowed = set(schema["$defs"]["starter_entry"]["properties"])
    card = _valid_card()
    for side in ("home", "away"):
        assert set(card["starters"][side]).issubset(allowed)
