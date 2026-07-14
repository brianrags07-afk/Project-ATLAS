# ATLAS Brain Phase 2C Complete

## Status

Phase 2C is frozen and complete.

Phase 2C establishes the canonical scoring-state timeline for
the verified 2024 regular-season game set.

## Frozen Engines

- `atlas/game_intelligence/scoring_state_timeline.py`
- `atlas/game_intelligence/scoring_timeline_season_builder.py`

## Frozen Artifacts

- `data/game_intelligence/scoring_timelines/2024/scoring_state_timelines.parquet`
- `data/game_intelligence/scoring_timelines/2024/scoring_state_timeline_audit.parquet`
- `data/game_intelligence/scoring_timelines/2024/scoring_state_timeline_failures.parquet`
- `data/game_intelligence/scoring_timelines/2024/scoring_state_timeline_metadata.json`

## Completion Metrics

- Verified games: 2,428
- Games represented: 2,428
- Scoring-state transitions: 16,450
- Audit rows: 2,428
- Audit failures: 0
- Build failures: 0
- Duplicate scoring-event rows: 0
- Raw score-state repairs preserved: 54
- Within-plate-appearance score changes: 114
- Delayed source-score updates: 1
- Canonical attribution repairs: 1
- Canonical batting/scoring-side mismatches: 0
- Regression tests passed: 17

## Architectural Rules Frozen

1. The scoring timeline is built from every ordered event row
   that changes the score.

2. Scoring is not limited to terminal plate-appearance rows.

3. Canonical pre-score state is carried forward from the prior
   verified scoring transition.

4. Raw source score states are preserved for provenance.

5. Score changes occurring multiple times during one plate
   appearance are represented as separate transitions.

6. Delayed Statcast score updates are explicitly classified.

7. The raw source inning and half are retained when canonical
   attribution is repaired.

8. Canonical batting side must equal the scoring side.

9. Every game contains exactly one terminal scoring transition.

10. Every timeline final score must match the frozen Phase 2A
    game outcome.

11. Phase 2C creates no predictions.

12. Phase 2C modifies no identities.

13. Phase 2C creates no game explanations.

14. Phase 2C uses no future games.

## Verified Edge Cases

- Ordinary road shutout: game `744795`
- Extra-inning one-run walkoff: game `745039`
- Multi-run walkoff: game `746576`
- Delayed score-update attribution: game `747004`
- Raw score-state discrepancies
- Multiple score changes within one plate appearance
- Continuous score progression across every verified game

## Regression Test File

- `tests/test_scoring_state_timeline.py`

## Freeze Rule

Future phases may consume Phase 2C artifacts but must not
silently alter Phase 2C definitions or outputs.

Any future Phase 2C change requires:

1. A documented defect.
2. A reproducible failing game.
3. A backward-compatible migration plan or explicit version bump.
4. Full 2024 regression testing.
5. A new milestone commit.
