# ATLAS Brain Phase 2E Working Checkpoint

> **Superseded:** This checkpoint only covers progress through Phase 2E.3A
> and is retained for historical context. Phases 2E.3B through 2E.4G have
> since completed and been frozen (see
> `docs/ATLAS_BRAIN_PHASE_2E_4F_TO_4G_CHECKPOINT.md` and
> `docs/ATLAS_BRAIN_PHASE_2E_4G_IMMUTABLE_CONCEPT_FREEZE.md`). The current
> open phase is 2E.5A (see
> `docs/ATLAS_BRAIN_PHASE_2E_5A_2025_VALIDATION_INPUT_READINESS.md`). For the
> authoritative, continuously updated status, see
> `docs/AUTOPILOT_EXECUTION_LEDGER.md`.

## Status

Phase 2E is **working and artifact-complete through Phase 2E.3A**, but it is
not yet frozen as a permanent source-code milestone.

The current notebook implementation has successfully produced leakage-safe
pregame team identities and team-versus-opponent matchup edges. The next
required engineering step is to convert the successful notebook logic into
permanent modules under `atlas/`, add regression tests, rerun the artifacts,
and then freeze and commit Phase 2E.

## Completed work

### Phase 2E.1 — Pregame identity source registry

- Frozen Phase 2D source rows: 4,856
- Frozen Phase 2D source columns: 121
- Approved lagged identity sources: 87
- Same-game identity sources: 0
- Future games used: no

### Phase 2E.2 — Strict prior-date team identity timeline

- Team-game rows: 4,856
- Unique games: 2,428
- Teams: 30
- Identity features: 87
- Date-team audit rows: 4,798
- Audit failures: 0
- Saved failure rows: 0
- Same-date games used: no
- Future games used: no

### Phase 2E.3A — Team-versus-opponent identity matchups

- Matchup rows: 4,856
- Unique games: 2,428
- Team identity columns: 87
- Opponent identity columns: 87
- Raw identity edges: 87
- Absolute identity edges: 87
- Mirror failures: 0
- Same-date games used: no
- Future games used: no

## Saved artifacts

- `data/game_intelligence/pregame_identity_registry/2024/pregame_identity_source_registry.csv`
- `data/game_intelligence/pregame_identity_registry/2024/pregame_identity_source_registry_metadata.json`
- `data/game_intelligence/pregame_team_identities/2024/pregame_team_identity_timeline.parquet`
- `data/game_intelligence/pregame_team_identities/2024/pregame_team_identity_timeline_audit.parquet`
- `data/game_intelligence/pregame_team_identities/2024/pregame_team_identity_timeline_failures.parquet`
- `data/game_intelligence/pregame_team_identities/2024/pregame_team_identity_timeline_metadata.json`
- `data/game_intelligence/pregame_identity_matchups/2024/pregame_identity_matchups.parquet`
- `data/game_intelligence/pregame_identity_matchups/2024/pregame_identity_matchups_metadata.json`

## Important limitation

The Phase 2E artifacts are saved in Drive, but the successful Phase 2E
builder logic has not yet been converted into permanent tracked Python
modules and regression tests. Therefore Phase 2E must not yet be described
as frozen.

## Required freeze work

1. Create permanent Phase 2E source modules under `atlas/game_intelligence/`.
2. Add regression tests covering registry, chronology, doubleheader safety,
   opponent mirrors, feature counts and metadata.
3. Rebuild Phase 2E from those modules.
4. Run all Phase 2A–2E regression tests.
5. Create `ATLAS_BRAIN_PHASE_2E_COMPLETE.md`.
6. Commit and push the Phase 2E milestone.

## Architecture direction

Phase 2E is the leakage-safe Brain layer that should connect existing
starter, bullpen, lineup, environment, series, totals, margin, prediction,
player-prop and explanation engines. Existing engines should be inspected
and connected before replacements are built.
