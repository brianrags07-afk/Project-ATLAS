"""
Tests for the redesigned atlas/audit/coverage_matrix.py and
atlas/audit/readiness.py: independent evidence dimensions, ATLAS
no-leakage rules, and per-decision required-dimension logic.
"""

from __future__ import annotations

from atlas.audit.coverage_matrix import DIMENSION_KEYS, build_coverage_matrix
from atlas.audit.evidence import DIMENSION_VALUES
from atlas.audit.provenance import build_all_dataset_provenance
from atlas.audit.readiness import ALLOWED_VERDICTS, DECISION_KEYS, build_readiness_decisions

REPO_INVENTORY_FIXTURE = {
    "focus_area_index": {
        "memories": ["atlas/memory/team_memory.py"],
        "identities": ["atlas/identities/identity_engine.py"],
        "concepts": [],
        "validation": ["atlas/validation/validation_engine.py"],
        "prediction": ["atlas/predictions/prediction_engine.py"],
        "pregame_snapshots": ["atlas/pregame/canonical_core_evidence_matrix.py"],
    }
}

EMPTY_REPO_INVENTORY = {"focus_area_index": {}}


def _dataset_profiles_fixture():
    return {
        "master_game_database": {
            "cloud_path": "data/master/master_game_database.parquet",
            "rows_by_season": {"2024": 100, "2025": 50},
            "unique_games_by_season": {"2024": 100, "2025": 50},
            "feature_presence": {
                "final_outcomes": "home_score",
                "starter_information": None,
                "bullpen_usage": None,
                "lineups": None,
                "injuries": None,
                "weather": None,
                "venue": "venue",
                "umpire": None,
                "rest": None,
                "travel": None,
                "published_series_context": None,
                "market_data": None,
            },
            "column_classification": {
                "home_score": "postgame_fact",
                "venue": "schedule_safe",
            },
            "null_percentages": {"home_score": 0.0, "venue": 0.0},
            "schema_fingerprint": "game-fp-1",
            "data_layer": "normalized_master",
        },
        "master_pitch_database": {
            "cloud_path": "data/master/master_pitch_database.parquet",
            "rows_by_season": {"2024": 1000},
            "pitches_by_season": {"2024": 1000},
            "unique_games_by_season": {"2024": 100},
            "schema_fingerprint": "pitch-fp-1",
            "data_layer": "normalized_master",
        },
        # An independent expected-game-count reference (e.g. a published
        # season schedule / league game-count registry), deliberately
        # distinct from master_game_database's own observed
        # unique_games_by_season. final_scores completeness must never
        # compare master_game_database against itself.
        "season_schedule": {
            "expected_games_by_season": {"2024": 100, "2025": 50},
        },
    }


def _matrix_row(matrix, row, season):
    return next(r for r in matrix if r["row"] == row and r["season"] == season)


# --------------------------------------------------------------------------
# Coverage matrix: independent dimensions.
# --------------------------------------------------------------------------


def test_coverage_matrix_rows_use_only_allowed_dimension_values():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    for row in matrix:
        for key in DIMENSION_KEYS:
            assert row[key] in DIMENSION_VALUES[key], (row["row"], row["season"], key, row[key])
        assert "evidence" in row and isinstance(row["evidence"], list)
        assert "risks" in row and isinstance(row["risks"], list)
        assert "required_next_evidence" in row and isinstance(row["required_next_evidence"], list)


def test_independent_dimensions_do_not_overwrite_each_other():
    """A row where data_presence=present and source_completeness=complete
    must still be able to have pregame_safety=unsafe -- the dimensions are
    independent, not derived from one collapsed status.

    "complete" here is proven by null_percentages == 0% (populated data)
    *and* the fixture's independent "season_schedule" expected-game-count
    reference matching observed unique games -- not by 0% nulls alone,
    which would only prove the score field is populated for whichever
    games happen to be present."""
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "complete"
    assert row["pregame_safety"] == "unsafe"
    assert row["temporal_availability"] == "postgame_only"


def test_complete_and_postgame_only_valid_for_reconstruction_but_unsafe_pregame():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "pitch_by_pitch", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "complete"
    assert row["temporal_availability"] == "postgame_only"
    assert row["pregame_safety"] == "unsafe"
    assert any("historical reconstruction" in r for r in row["risks"])


def test_final_scores_evidenced_by_game_database_not_pitch_database():
    """final_scores must be evidenced by master_game_database's
    final-outcome columns only. Removing master_pitch_database entirely
    must not change the final_scores row at all.

    "complete" is proven by both null_percentages == 0% and the fixture's
    independent "season_schedule" expected-game-count reference matching
    observed unique games -- 0% nulls alone is not sufficient evidence of
    full-season coverage."""
    profiles = _dataset_profiles_fixture()
    matrix_with_pitch = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row_with_pitch = _matrix_row(matrix_with_pitch, "final_scores", 2024)

    profiles_without_pitch = _dataset_profiles_fixture()
    del profiles_without_pitch["master_pitch_database"]
    matrix_without_pitch = build_coverage_matrix(profiles_without_pitch, REPO_INVENTORY_FIXTURE)
    row_without_pitch = _matrix_row(matrix_without_pitch, "final_scores", 2024)

    for key in DIMENSION_KEYS:
        assert row_with_pitch[key] == row_without_pitch[key]
    assert row_without_pitch["data_presence"] == "present"
    assert row_without_pitch["source_completeness"] == "complete"

    # Conversely, removing master_game_database (final_scores' real
    # source) must make final_scores missing even though
    # master_pitch_database is still fully populated.
    profiles_without_game_db = _dataset_profiles_fixture()
    del profiles_without_game_db["master_game_database"]
    matrix_without_game_db = build_coverage_matrix(profiles_without_game_db, REPO_INVENTORY_FIXTURE)
    row_without_game_db = _matrix_row(matrix_without_game_db, "final_scores", 2024)
    assert row_without_game_db["data_presence"] == "missing"


def test_single_pitch_row_representing_one_game_cannot_be_complete():
    """A single observed game in master_pitch_database, when
    master_game_database expects many games that season, must be
    'partial', never 'complete'. Row/game presence alone is not evidence
    of complete season coverage."""
    profiles = _dataset_profiles_fixture()
    profiles["master_pitch_database"]["pitches_by_season"] = {"2024": 1}
    profiles["master_pitch_database"]["rows_by_season"] = {"2024": 1}
    profiles["master_pitch_database"]["unique_games_by_season"] = {"2024": 1}
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "pitch_by_pitch", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "partial"
    assert row["source_completeness"] != "complete"


def test_pitch_level_completeness_unknown_without_expected_game_reference():
    """When master_game_database's expected-game-count reference is
    unavailable, pitch-level completeness must stay 'unknown' rather than
    being inferred 'complete' from row presence."""
    profiles = _dataset_profiles_fixture()
    del profiles["master_game_database"]
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "pitch_by_pitch", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "unknown"


def _single_game_final_score_profile(expected_games=None):
    """A dataset profile with exactly one fully-populated final-score game
    in master_game_database, optionally paired with an independent
    "season_schedule" expected-game-count reference."""
    profiles = {
        "master_game_database": {
            "cloud_path": "data/master/master_game_database.parquet",
            "rows_by_season": {"2024": 1},
            "unique_games_by_season": {"2024": 1},
            "feature_presence": {"final_outcomes": "home_score"},
            "column_classification": {"home_score": "postgame_fact"},
            "null_percentages": {"home_score": 0.0},
            "schema_fingerprint": "game-fp-1",
            "data_layer": "normalized_master",
        }
    }
    if expected_games is not None:
        profiles["season_schedule"] = {"expected_games_by_season": {"2024": expected_games}}
    return profiles


def test_one_fully_populated_game_with_no_expected_season_reference_is_unknown():
    """One fully-populated (0% null) final-score game, with no independent
    expected-season-game-count reference available, must be 'unknown' --
    never 'complete'. 0% nulls only proves the score field is populated
    for the one game present; it says nothing about full-season
    coverage."""
    profiles = _single_game_final_score_profile(expected_games=None)
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "unknown"


def test_one_fully_populated_game_against_expected_count_of_100_is_partial():
    """One fully-populated final-score game, compared against an
    independent expected-game-count reference of 100 for the season, must
    be 'partial' -- a single observed game can never satisfy an expected
    count of 100."""
    profiles = _single_game_final_score_profile(expected_games=100)
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "partial"


def test_final_scores_complete_when_observed_games_meet_explicit_expected_count():
    """Fully-populated final scores with observed unique games equal to an
    explicit, independent expected-game-count reference must be
    'complete'."""
    profiles = _single_game_final_score_profile(expected_games=1)
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "complete"


def test_null_final_scores_stay_partial_even_when_game_coverage_is_complete():
    """Missing/null final-score values must remain 'partial' even when
    observed unique games meet the independent expected-game-count
    reference -- game coverage being complete does not excuse missing
    values."""
    profiles = _single_game_final_score_profile(expected_games=1)
    profiles["master_game_database"]["null_percentages"]["home_score"] = 50.0
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "partial"


def test_single_final_score_row_with_partial_nulls_is_not_complete():
    """A season where the final-outcome column is only partially populated
    (nonzero null percentage) must be 'partial', never 'complete'."""
    profiles = _dataset_profiles_fixture()
    profiles["master_game_database"]["null_percentages"]["home_score"] = 25.0
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "partial"


def test_processed_master_tables_do_not_prove_raw_source_readiness():
    """master_game_database/master_pitch_database are classified
    normalized_master, never raw_source, and this classification alone
    must not make any row's provenance_status 'verified'."""
    profiles = _dataset_profiles_fixture()
    assert profiles["master_game_database"]["data_layer"] == "normalized_master"
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "final_scores", 2024)
    assert row["provenance_status"] != "verified"


def test_completed_game_tables_do_not_prove_published_schedule_provenance():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "published_schedule", 2024)
    assert row["data_presence"] == "present"
    # Row presence in master_game_database is not proof of complete season
    # coverage: there is no independent expected-game-count reference for
    # the schedule itself, so completeness must stay "unknown" rather than
    # being inferred from row_count > 0.
    assert row["source_completeness"] == "unknown"
    assert row["provenance_status"] != "verified"
    assert row["pregame_safety"] == "unsafe"


def test_published_schedule_missing_for_season_with_no_rows():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "published_schedule", 2026)
    assert row["data_presence"] == "missing"


def test_series_context_inferred_from_results_is_unsafe():
    profiles = _dataset_profiles_fixture()
    profiles["master_game_database"]["feature_presence"]["published_series_context"] = "series_length"
    profiles["master_game_database"]["column_classification"]["series_length"] = "schedule_safe"
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "published_series_context", 2024)
    assert row["data_presence"] == "present"
    assert row["pregame_safety"] == "unsafe"
    assert row["temporal_availability"] == "postgame_only"


def test_series_context_row_presence_alone_cannot_be_complete():
    """published_series_context has no independent expected-count
    reference in this audit; row presence must yield 'unknown' source
    completeness, never 'complete'."""
    profiles = _dataset_profiles_fixture()
    profiles["master_game_database"]["feature_presence"]["published_series_context"] = "series_length"
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "published_series_context", 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "unknown"
    assert row["source_completeness"] != "complete"


def test_dynamic_pregame_field_missing_evidence_stays_unknown_not_unsafe():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    row = _matrix_row(matrix, "starters", 2024)
    assert row["data_presence"] == "missing"
    assert row["pregame_safety"] in ("unknown", "not_applicable")


def test_concept_discovery_missing_when_no_modules():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), EMPTY_REPO_INVENTORY)
    row = _matrix_row(matrix, "concept_discovery", 2024)
    assert row["data_presence"] == "missing"


def test_unknown_evidence_remains_unknown_not_promoted():
    """A row with no feature-presence mapping (e.g. an unmapped dataset)
    must remain unknown rather than being promoted to a positive value."""
    matrix = build_coverage_matrix({}, EMPTY_REPO_INVENTORY)
    row = _matrix_row(matrix, "weather", 2024)
    assert row["data_presence"] == "missing"
    assert row["pregame_safety"] in ("unknown", "not_applicable")
    row2 = _matrix_row(matrix, "starters", 2025)
    assert row2["provenance_status"] in ("missing", "unknown")


# --------------------------------------------------------------------------
# Readiness decisions A-G.
# --------------------------------------------------------------------------


def test_readiness_decisions_cover_all_seven_questions_with_full_contract():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    decisions = readiness["decisions"]
    assert set(decisions.keys()) == set(DECISION_KEYS)
    for entry in decisions.values():
        assert entry["verdict"] in ALLOWED_VERDICTS
        assert "required_dimensions" in entry and entry["required_dimensions"]
        assert "evidence_used" in entry
        assert "blockers" in entry
        assert "warnings" in entry
        assert "next_action" in entry and entry["next_action"]
        assert "does_not_authorize" in entry and entry["does_not_authorize"]


def test_exact_reproduction_blocked_without_manifests_and_hashes():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    profiles = _dataset_profiles_fixture()
    provenance = build_all_dataset_provenance(profiles, cloud_inventory=None)
    readiness = build_readiness_decisions(matrix, profiles, provenance)
    decision = readiness["decisions"]["A_exact_2024_reproduction"]
    assert decision["verdict"] == "not_ready"
    assert decision["blockers"]


def test_exact_reproduction_ready_when_hash_and_manifest_verified():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    profiles = _dataset_profiles_fixture()
    cloud_inventory = {
        "objects": [
            {"full_path": "gs://bucket/data/master/master_game_database.parquet", "md5_hash": "abc"},
            {"full_path": "gs://bucket/data/master/master_pitch_database.parquet", "md5_hash": "def"},
        ]
    }
    provenance = build_all_dataset_provenance(
        profiles,
        cloud_inventory=cloud_inventory,
        manifest_linkages={"master_game_database": "manifest-1", "master_pitch_database": "manifest-1"},
    )
    assert provenance["master_game_database"]["provenance_status"] == "verified"
    readiness = build_readiness_decisions(matrix, profiles, provenance)
    decision = readiness["decisions"]["A_exact_2024_reproduction"]
    assert decision["verdict"] != "not_ready" or "concept-discovery" in " ".join(decision["blockers"])


def test_rebuild_2024_ready_with_verified_raw_provenance_despite_postgame_facts():
    """A 2024 rebuild can be ready with postgame raw facts when raw
    provenance and completeness are verified -- postgame-only status must
    not block reconstruction readiness."""
    profiles = _dataset_profiles_fixture()
    profiles["raw_pitch_feed"] = {
        "cloud_path": "data/raw/statsapi/pitch_feed_2024.parquet",
        "rows_by_season": {"2024": 5000},
        "row_count": 5000,
        "data_layer": "raw_source",
        "schema_fingerprint": "raw-fp-1",
    }
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, profiles)
    decision = readiness["decisions"]["B_rebuild_2024_from_raw"]
    assert not decision["blockers"]
    assert decision["verdict"] in ("ready", "ready_with_warnings")


def test_rebuild_2024_warns_without_any_raw_source_evidence():
    profiles = _dataset_profiles_fixture()
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, profiles)
    decision = readiness["decisions"]["B_rebuild_2024_from_raw"]
    assert any("raw_source" in w for w in decision["warnings"])


def test_schema_incompatibility_blocks_identical_2025_transformations():
    profiles = _dataset_profiles_fixture()
    reference = dict(profiles["master_game_database"])
    reference["dtypes"] = {"home_score": "int64", "venue": "object"}
    candidate = dict(profiles["master_game_database"])
    candidate["dtypes"] = {"home_score": "float64", "venue": "object", "new_column": "object"}
    profiles["master_game_database__2024_reference"] = reference
    profiles["master_game_database"] = candidate
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, profiles)
    decision = readiness["decisions"]["D_parse_2025_identical_transformations"]
    assert decision["verdict"] == "not_ready"
    assert any("incompatible" in b for b in decision["blockers"])


def test_missing_historical_timestamps_block_2025_walk_forward_backtest():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    decision = readiness["decisions"]["E_2025_walk_forward_backtest"]
    assert decision["verdict"] == "not_ready"
    assert any("timestamp" in b or "leak" in b for b in decision["blockers"])


def test_leakage_guard_rejects_postgame_only_field_for_backtest():
    profiles = _dataset_profiles_fixture()
    profiles["master_game_database"]["feature_presence"]["starter_information"] = "starter_name"
    profiles["master_game_database"]["column_classification"]["starter_name"] = "postgame_fact"
    profiles["master_game_database"]["null_percentages"]["starter_name"] = 0.0
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    starters_row = next(r for r in matrix if r["row"] == "starters" and r["season"] == 2025)
    assert starters_row["pregame_safety"] == "unsafe"
    readiness = build_readiness_decisions(matrix, profiles)
    decision = readiness["decisions"]["E_2025_walk_forward_backtest"]
    assert decision["verdict"] == "not_ready"
    assert any("leak" in b for b in decision["blockers"])


def test_pregame_game_cards_ready_with_timestamp_proven_snapshot():
    profiles = _dataset_profiles_fixture()
    matrix = build_coverage_matrix(profiles, REPO_INVENTORY_FIXTURE)
    for row in matrix:
        if row["row"] in ("starters", "lineups", "bullpen_usage", "weather", "rest", "travel") and row["season"] == 2025:
            row["data_presence"] = "present"
            row["pregame_safety"] = "safe"
    readiness = build_readiness_decisions(matrix, profiles)
    decision = readiness["decisions"]["F_2025_pregame_game_cards"]
    assert not decision["blockers"]


def test_forward_predictions_2026_not_ready_without_verified_schedule():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    decision = readiness["decisions"]["G_2026_forward_predictions"]
    assert decision["verdict"] == "not_ready"


def test_readiness_note_states_full_table_is_not_automatically_pregame_safe():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    assert "not automatically pregame-safe" in readiness["note"]


def test_every_decision_states_baseline_non_authorizations():
    matrix = build_coverage_matrix(_dataset_profiles_fixture(), REPO_INVENTORY_FIXTURE)
    readiness = build_readiness_decisions(matrix, _dataset_profiles_fixture())
    for decision in readiness["decisions"].values():
        joined = " ".join(decision["does_not_authorize"])
        assert "rebuild" in joined
        assert "backtest" in joined
        assert "training" in joined
        assert "prediction" in joined
