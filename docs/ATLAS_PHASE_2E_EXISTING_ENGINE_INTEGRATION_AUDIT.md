# ATLAS Phase 2E Existing Engine Integration Audit

## Purpose

Identify existing tracked engines that should connect to the leakage-safe Phase 2E Brain before new prediction infrastructure is built.

## Summary

- Tracked ATLAS source modules inspected: 116
- Tracked test modules inspected: 5
- Required engine families: 14
- Engine families detected: 14
- Strong connection candidates: 6
- Validation-first candidates: 4

## Recommended integration candidates

### pregame_prediction

- Status: `STRONG_CONNECT_CANDIDATE`
- Recommended module: `atlas/game_intelligence/reconstruction.py`
- Candidate modules detected: 29
- Direct mapped tests: 3
- Public functions: 2
- Pregame-safety signal: True
- Action: Inspect function contracts, run a focused smoke test, then connect to Phase 2E.

### run_line_margin

- Status: `INTERFACE_EXISTS_REVIEW_SAFETY`
- Recommended module: `atlas/game_intelligence/outcome_classifier.py`
- Candidate modules detected: 12
- Direct mapped tests: 2
- Public functions: 2
- Pregame-safety signal: False
- Action: Verify chronology, inputs and output semantics before integration.

### totals_scoring

- Status: `INTERFACE_EXISTS_REVIEW_SAFETY`
- Recommended module: `atlas/game_intelligence/outcome_classifier.py`
- Candidate modules detected: 9
- Direct mapped tests: 2
- Public functions: 2
- Pregame-safety signal: False
- Action: Verify chronology, inputs and output semantics before integration.

### starter_pitching

- Status: `STRONG_CONNECT_CANDIDATE`
- Recommended module: `atlas/game_intelligence/contracts.py`
- Candidate modules detected: 33
- Direct mapped tests: 1
- Public functions: 3
- Pregame-safety signal: True
- Action: Inspect function contracts, run a focused smoke test, then connect to Phase 2E.

### bullpen_identity

- Status: `STRONG_CONNECT_CANDIDATE`
- Recommended module: `atlas/game_intelligence/reconstruction.py`
- Candidate modules detected: 23
- Direct mapped tests: 3
- Public functions: 2
- Pregame-safety signal: True
- Action: Inspect function contracts, run a focused smoke test, then connect to Phase 2E.

### bullpen_availability_fatigue

- Status: `VALIDATE_CONNECT_CANDIDATE`
- Recommended module: `atlas/identities/bullpen_availability_fatigue_engine.py`
- Candidate modules detected: 7
- Direct mapped tests: 0
- Public functions: 1
- Pregame-safety signal: True
- Action: Add focused validation because direct test coverage was not detected.

### lineup_matchup

- Status: `STRONG_CONNECT_CANDIDATE`
- Recommended module: `atlas/game_intelligence/reconstruction.py`
- Candidate modules detected: 47
- Direct mapped tests: 3
- Public functions: 2
- Pregame-safety signal: True
- Action: Inspect function contracts, run a focused smoke test, then connect to Phase 2E.

### lineup_pitch_compatibility

- Status: `VALIDATE_CONNECT_CANDIDATE`
- Recommended module: `atlas/interactions/walk_forward_snapshot_engine.py`
- Candidate modules detected: 8
- Direct mapped tests: 0
- Public functions: 5
- Pregame-safety signal: True
- Action: Add focused validation because direct test coverage was not detected.

### environment_park

- Status: `VALIDATE_CONNECT_CANDIDATE`
- Recommended module: `atlas/identities/bullpen_availability_fatigue_engine.py`
- Candidate modules detected: 13
- Direct mapped tests: 0
- Public functions: 1
- Pregame-safety signal: True
- Action: Add focused validation because direct test coverage was not detected.

### series_rest_travel

- Status: `INTERFACE_EXISTS_REVIEW_SAFETY`
- Recommended module: `atlas/game_intelligence/outcome_classifier.py`
- Candidate modules detected: 33
- Direct mapped tests: 2
- Public functions: 2
- Pregame-safety signal: False
- Action: Verify chronology, inputs and output semantics before integration.

### player_props

- Status: `INTERFACE_EXISTS_REVIEW_SAFETY`
- Recommended module: `atlas/game_intelligence/outcome_classifier.py`
- Candidate modules detected: 20
- Direct mapped tests: 2
- Public functions: 2
- Pregame-safety signal: False
- Action: Verify chronology, inputs and output semantics before integration.

### calibration

- Status: `VALIDATE_CONNECT_CANDIDATE`
- Recommended module: `atlas/learning/league_evidence_discovery.py`
- Candidate modules detected: 6
- Direct mapped tests: 0
- Public functions: 3
- Pregame-safety signal: True
- Action: Add focused validation because direct test coverage was not detected.

### backtest_validation

- Status: `STRONG_CONNECT_CANDIDATE`
- Recommended module: `atlas/game_intelligence/reconstruction.py`
- Candidate modules detected: 46
- Direct mapped tests: 3
- Public functions: 2
- Pregame-safety signal: True
- Action: Inspect function contracts, run a focused smoke test, then connect to Phase 2E.

### evidence_explanation

- Status: `STRONG_CONNECT_CANDIDATE`
- Recommended module: `atlas/game_intelligence/reconstruction.py`
- Candidate modules detected: 58
- Direct mapped tests: 3
- Public functions: 2
- Pregame-safety signal: True
- Action: Inspect function contracts, run a focused smoke test, then connect to Phase 2E.

## Architecture decision

Existing starter, bullpen, lineup, environment, margin, totals, prediction, calibration, player-prop and explanation engines must be validated and connected to Phase 2E before replacements are considered.

Phase 2E remains a working checkpoint until its notebook logic is converted into permanent modules, regression-tested and frozen.