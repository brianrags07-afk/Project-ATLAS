# ATLAS Brain Phase 2E.3B — Prediction Logic Governance

## Purpose

Prevent hand-authored baseball beliefs, fixed prediction weights and arbitrary
confidence adjustments from entering the new ATLAS Brain.

## Governance principle

ATLAS may be given:

- factual historical outcome definitions;
- raw pregame measurements;
- chronology and leakage controls;
- statistical reliability and validation controls.

ATLAS may not be given:

- manually assigned baseball feature weights;
- fixed home, park, bullpen or lineup bonuses;
- handcrafted fatigue or recovery scores as predictive truth;
- forced feature directions;
- arbitrary confidence or probability adjustments;
- legacy fusion rules that have not been learned from training data.

## Module decisions

### Quarantined legacy decision engines

- `atlas/predictions/pregame_prediction_engine.py`
- `atlas/predictions/prediction_fusion_engine.py`
- `atlas/backtest/weighted_state_backtest_engine.py`

These modules remain preserved for historical comparison but cannot power the
Phase 2E Brain or production predictions.

### Raw-facts-only bullpen modules

- `atlas/identities/bullpen_availability_fatigue_engine.py`
- `atlas/identities/bullpen_identity_integration_engine.py`

Allowed fields include workload, appearances, pitches, days of rest,
back-to-back usage, three-in-four usage and reliever availability facts.

Handcrafted pressure, fatigue, recovery and availability scores are not
approved as predictive inputs.

### Reliability-only discovery modules

- `atlas/learning/team_evidence_discovery.py`
- `atlas/learning/league_evidence_discovery.py`
- `atlas/learning/bullpen_evidence_discovery_2024.py`

These modules may measure sample size, effect size, significance and
validation stability. Their manually constructed confidence scores cannot
determine outcome probability or predictive influence.

## Saved governance artifacts

- `data/governance/prediction_logic/prediction_logic_adjudication_registry.csv`
- `data/governance/prediction_logic/prediction_module_policy.csv`
- `data/governance/prediction_logic/prediction_logic_policy_metadata.json`

## Permanent code

- `atlas/governance/prediction_logic_policy.py`
- `tests/test_prediction_logic_policy.py`

## Next engineering step

Build clean Phase 2E learning inputs from:

1. leakage-safe identity and matchup edges;
2. raw starter and lineup interaction facts;
3. raw bullpen workload and availability facts;
4. raw park, weather, series, rest and travel facts;
5. factual historical targets.

Predictive relationships and feature influence will then be learned from the
2024 training season and validated independently on 2025.
