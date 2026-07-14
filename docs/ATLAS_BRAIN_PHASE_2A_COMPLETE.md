# ATLAS Baseball Brain — Phase 2A Complete

Completion date: 2026-07-14T03:36:20.488896+00:00

## Phase

Deterministic factual game-outcome classification.

## Question answered

Phase 2A answers:

> What factually happened in the completed game?

It does not explain why the game happened, update identities,
discover concepts, assign prediction weights, or create predictions.

## Frozen input boundary

Phase 2A accepts only a verified Phase 1 game reconstruction.

The classifier uses:

- corrected canonical final scores
- complete pitch-event sequence
- plate-appearance terminal score states
- inning and half-inning state
- Phase 1 reconstruction verification

## Canonical 2024 output

- Eligible games classified: 2,428
- Known source exceptions excluded: 1
- Duplicate game IDs: 0
- Build failures: 0
- Outcome audit failures: 0
- Score-verification failures: 0
- Reconstruction-verification failures: 0
- Regression tests passed: 11

## Factual labels produced

Phase 2A produces game-level facts including:

- winner and loser
- home and away result
- final score
- total runs
- signed and absolute run margin
- one-run result
- margins beyond 1.5, 3.5 and 5.5
- shutout outcome
- fixed total-run thresholds
- innings played
- extra innings
- tied after regulation
- walk-off result
- terminal scoring play
- comeback win
- lead changes
- tied-state entries
- largest home and away leads
- first scoring side
- scoreless-through-inning labels
- early, middle and late scoring totals
- scoring plays
- event and plate-appearance counts
- provenance and safety flags

## Independently verified logic

The 2024 event store was independently reduced to one terminal
score state per plate appearance.

The independent audit found:

- comeback mismatches: 0
- lead-change mismatches: 0
- tied-state mismatches: 0
- comeback/lead-change equivalence exceptions: 0

Under the frozen definitions:

`comeback_win == (lead_changes > 0)`

A comeback win means the eventual winner trailed at least once.
A lead change means the non-tied game leader changed teams.

## Known 2024 distributions

- Home wins: 1,267
- Away wins: 1,161
- One-run games: 675
- Shutouts: 321
- Extra-inning games: 216
- Walk-offs: 208
- Comeback wins: 1,026

These counts are regression-protected.

## Freeze rule

Phase 2A may be changed only to correct a verified factual defect.

Later modules may consume Phase 2A outputs but may not insert
explanation, identity, learning, prediction or market logic into the
outcome classifier.

## Next phase

Phase 2B: team-perspective factual outcomes.

Phase 2B will convert each game-level outcome into two factual rows:

- one home-team perspective
- one away-team perspective

No identity updating or explanation will occur in Phase 2B.
