# ATLAS Tracked Engine Architecture Audit

## Purpose

This audit maps only GitHub-tracked source code. It is intended to prevent
rebuilding engines that already exist and to identify which existing modules
should be connected to the Phase 2E Brain.

## Repository status

- Tracked files: 149
- Tracked Python files: 123
- Tracked source modules: 118
- Tracked test modules: 5
- Source parse failures: 0
- Modules with detected direct tests: 7

## Phase 2E status

Phase 2E is preserved as a working checkpoint through Phase 2E.3A.
Its artifacts pass all saved gates, but its notebook logic must still be
converted into permanent source modules and regression tests before freezing.

## Engine recommendation categories

- `USE_OR_CONNECT`: tested prediction-critical implementation candidate.
- `VALIDATE_THEN_CONNECT`: broad implementation candidate needing behavior validation.
- `INSPECT_AND_VALIDATE`: focused implementation without detected direct tests.
- `SUPPORTING_MODULE`: reusable infrastructure, not necessarily a prediction engine.
- `PACKAGE_SUPPORT`: package initializer.
- `LEGACY_REVIEW`: possible older or superseded implementation.
- `REPAIR_NOW`: parsing or syntax failure.

## Leading canonical candidates

## prediction_model
- `atlas/game_intelligence/reconstruction.py` (rank 1, tests 3, functions 18, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/outcome_classifier.py` (rank 2, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 3, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/history/failure_analysis_engine.py` (rank 4, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 5, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)

## run_line_margin
- `atlas/game_intelligence/outcome_classifier.py` (rank 1, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/team_outcome_classifier.py` (rank 2, tests 1, functions 4, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/team_game_flow.py` (rank 3, tests 0, functions 10, recommendation VALIDATE_THEN_CONNECT)
- `atlas/interactions/lineup_starter_input_engine.py` (rank 4, tests 0, functions 11, recommendation VALIDATE_THEN_CONNECT)
- `atlas/game_intelligence/game_flow_fact_table.py` (rank 5, tests 0, functions 9, recommendation VALIDATE_THEN_CONNECT)

## totals_scoring
- `atlas/game_intelligence/outcome_classifier.py` (rank 1, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 2, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/team_outcome_classifier.py` (rank 3, tests 1, functions 4, recommendation USE_OR_CONNECT)
- `atlas/learning/team_evidence_discovery.py` (rank 4, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/game_intelligence/scoring_timeline_season_builder.py` (rank 5, tests 0, functions 8, recommendation VALIDATE_THEN_CONNECT)

## starter_pitching
- `atlas/history/failure_analysis_engine.py` (rank 1, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 2, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)
- `atlas/learning/team_evidence_discovery.py` (rank 3, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/learning/evidence_consolidation_engine.py` (rank 4, tests 0, functions 14, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_availability_fatigue_engine.py` (rank 5, tests 0, functions 14, recommendation VALIDATE_THEN_CONNECT)

## bullpen
- `atlas/game_intelligence/reconstruction.py` (rank 1, tests 3, functions 18, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/__init__.py` (rank 2, tests 3, functions 0, recommendation PACKAGE_SUPPORT)
- `atlas/history/failure_analysis_engine.py` (rank 3, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 4, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)
- `atlas/learning/evidence_consolidation_engine.py` (rank 5, tests 0, functions 14, recommendation VALIDATE_THEN_CONNECT)

## lineup_matchup
- `atlas/game_intelligence/reconstruction.py` (rank 1, tests 3, functions 18, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/__init__.py` (rank 2, tests 3, functions 0, recommendation PACKAGE_SUPPORT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 3, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/history/failure_analysis_engine.py` (rank 4, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 5, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)

## environment
- `atlas/history/failure_analysis_engine.py` (rank 1, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_availability_fatigue_engine.py` (rank 2, tests 0, functions 14, recommendation VALIDATE_THEN_CONNECT)
- `atlas/history/model_repair_planning_engine.py` (rank 3, tests 0, functions 12, recommendation VALIDATE_THEN_CONNECT)
- `atlas/questions/question_library.py` (rank 4, tests 0, functions 5, recommendation VALIDATE_THEN_CONNECT)
- `atlas/gamecards/gamecard_engine.py` (rank 5, tests 0, functions 5, recommendation VALIDATE_THEN_CONNECT)

## series_rest_travel
- `atlas/game_intelligence/outcome_classifier.py` (rank 1, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 2, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/history/failure_analysis_engine.py` (rank 3, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 4, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)
- `atlas/learning/team_evidence_discovery.py` (rank 5, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)

## player_props
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 1, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)
- `atlas/learning/evidence_consolidation_engine.py` (rank 2, tests 0, functions 14, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_availability_fatigue_engine.py` (rank 3, tests 0, functions 14, recommendation VALIDATE_THEN_CONNECT)
- `atlas/learning/bullpen_concept_consolidation_2024.py` (rank 4, tests 0, functions 9, recommendation VALIDATE_THEN_CONNECT)
- `atlas/pitchers/v2/summaries.py` (rank 5, tests 0, functions 11, recommendation VALIDATE_THEN_CONNECT)

## explanations
- `atlas/game_intelligence/reconstruction.py` (rank 1, tests 3, functions 18, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/outcome_classifier.py` (rank 2, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/__init__.py` (rank 3, tests 3, functions 0, recommendation PACKAGE_SUPPORT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 4, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/history/failure_analysis_engine.py` (rank 5, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)

## pregame_safety
- `atlas/game_intelligence/reconstruction.py` (rank 1, tests 3, functions 18, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/outcome_classifier.py` (rank 2, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/__init__.py` (rank 3, tests 3, functions 0, recommendation PACKAGE_SUPPORT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 4, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/history/failure_analysis_engine.py` (rank 5, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)

## team_identity
- `atlas/game_intelligence/reconstruction.py` (rank 1, tests 3, functions 18, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/outcome_classifier.py` (rank 2, tests 2, functions 15, recommendation USE_OR_CONNECT)
- `atlas/game_intelligence/scoring_state_timeline.py` (rank 3, tests 1, functions 9, recommendation USE_OR_CONNECT)
- `atlas/history/failure_analysis_engine.py` (rank 4, tests 0, functions 16, recommendation VALIDATE_THEN_CONNECT)
- `atlas/identities/bullpen_identity_integration_engine.py` (rank 5, tests 0, functions 13, recommendation VALIDATE_THEN_CONNECT)


## Required next actions

1. Inspect interfaces of the highest-ranked modules for winner, run-line,
   totals, starter, bullpen, lineup, environment, series/rest/travel,
   player props and explanations.
2. Identify canonical inputs and outputs for each engine.
3. Confirm pregame safety and season handling.
4. Connect approved engines to Phase 2E identities and matchup edges.
5. Convert Phase 2E notebook logic into permanent modules and tests.
6. Freeze and commit Phase 2E only after a complete rebuild and regression run.
