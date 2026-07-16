# ATLAS Brain Phase 2E.4F.2 — Concept Network Independence

## Status

Completed for the 2024 discovery universe.

## Purpose

Phase 2E.4F.1 removed concepts whose members came from the same broad evidence
domain or shared the same underlying metric.

Phase 2E.4F.2 identifies redundancy that remains across otherwise structurally
valid concepts.

## Input

- Phase 2E.4F.1 concepts: 5,046
- Targets: 2

## Network controls

Concepts are connected when one of the following is true:

1. They use the exact same pair of member conditions.
2. They share at least one member condition and have activation Jaccard
   similarity of at least 0.80.
3. They belong to the same target and broad-domain pair and have activation
   Jaccard similarity of at least 0.95.

Connected concepts form one explanatory network component.

## Frozen representative rule

Each component receives exactly one representative, selected in this order:

1. Strong concept status.
2. Lower discovery q-value.
3. Greater incremental lift.
4. Greater absolute joint lift.
5. Greater active sample.
6. Stable concept ID ordering.

These fields select a research representative only. They are not prediction
weights.

## Results

- Network edges: 1,268
- Network components: 4,271
- Frozen representatives: 4,271
- Maximum component size: 30

## Governance

- All Phase 2E.4F.1 concepts are preserved.
- Redundant concepts are labeled, not deleted.
- One representative exists per component.
- No market data was used.
- No prediction weight was assigned.
- No probability or prediction was created.
- No 2025 validation result was used.
- No 2026 result was used.

## Next phase

The frozen representative definitions may next be serialized with immutable
member features, operators and thresholds before blind application to the 2025
pregame evidence universe.
