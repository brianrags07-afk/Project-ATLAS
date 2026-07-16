# ATLAS Brain Phase 2E.3F — Starter and Lineup Missingness Audit

## Purpose

Determine what missing starter, pitcher, batter and lineup statistics mean
before any imputation or model training occurs.

## Governing rule

ATLAS does not automatically convert missing values to zero.

A missing value may represent:

- no prior MLB history,
- no qualifying pitch or batted-ball sample,
- an unpublished lineup,
- an unknown starter,
- an undefined statistical rate,
- or an upstream source failure.

These states must remain distinguishable.

## Scope

- Batter pregame snapshots
- Pitcher pregame snapshots
- Lineup-starter interaction inputs
- Seasons 2024, 2025 and 2026
- Historical coverage through July 3, 2026

## Results

- Critical existing key fields complete: True
- Artifact-season groups audited: 9
- Column-season combinations audited: 2,733
- Columns/seasons with at least 20% missingness: 0
- Values imputed: no
- Rows deleted: no
- Predictions created: no

## Highest-missing fields

No columns exceeded the 20% missingness threshold.

## Downstream policy

1. Keep undefined historical rates as missing.
2. Attach sample-size and availability indicators.
3. Separate measured zero from unknown.
4. Repair missing required join keys.
5. Exclude effectively unavailable fields until their source is repaired.
6. Fit any imputation values on training data only, never on validation or future data.
