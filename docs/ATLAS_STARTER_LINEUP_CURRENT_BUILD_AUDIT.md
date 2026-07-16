# ATLAS Starter and Lineup Current-Build Audit

## Purpose

Verify the already-existing starter, batter and lineup interaction pipeline rather than rebuilding it.

## Current-build modules

- `atlas/interactions/walk_forward_snapshot_engine.py`
- `atlas/interactions/pregame_snapshot_builder.py`
- `atlas/interactions/lineup_starter_input_engine.py`

## Canonical artifacts

- **batter**: /content/drive/MyDrive/Project_Atlas/data/pregame/snapshots/batter_pregame_snapshots.parquet — 124,457 rows, latest date 2026-07-03 00:00:00
- **pitcher**: /content/drive/MyDrive/Project_Atlas/data/pregame/snapshots/pitcher_pregame_snapshots.parquet — 52,689 rows, latest date 2026-07-03 00:00:00
- **lineup_starter**: /content/drive/MyDrive/Project_Atlas/data/pregame/interactions/lineup_starter_inputs.parquet — 12,350 rows, latest date 2026-07-03 00:00:00

## Completion status

REVIEW REQUIRED