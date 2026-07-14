# ATLAS Baseball Brain — Phase 2B Complete

Completion date: 2026-07-14T04:09:10.691047+00:00

## Phase

Team-perspective factual game outcomes.

## Question answered

Phase 2B answers:

> What factually happened to each team in the completed game?

Every eligible game produces exactly two immutable rows:

- one home-team perspective
- one away-team perspective

Phase 2B does not explain why the outcome happened, update
identities, discover evidence, assign weights, or create predictions.

## Frozen input boundary

Phase 2B consumes only the frozen Phase 2A game-outcome artifact.

It does not reread raw Statcast or independently derive final scores.

## Canonical 2024 output

- Games converted: 2,428
- Team-game rows: 4,856
- MLB teams represented: 30
- Duplicate team-games: 0
- Build failures: 0
- Game mirror-audit failures: 0
- Team-row audit failures: 0
- Regression tests passed: 13

## Team-perspective facts produced

Each team-game row contains:

- team and opponent
- home or away perspective
- team score and opponent score
- signed and absolute run differential
- win and loss result
- one-run result
- wins and losses beyond 1.5, 3.5 and 5.5
- scoring bands
- runs-allowed bands
- shutout win and shutout loss
- game total thresholds
- extra innings
- tied after regulation
- walk-off win and walk-off loss
- comeback win and comeback loss
- largest deficit overcome
- largest lead lost
- team and opponent early scoring
- team and opponent middle-inning scoring
- team and opponent late scoring
- first-scoring side
- scoreless-through-inning facts
- provenance and safety flags

## Mirror invariants

For every game:

- exactly two team rows exist
- one row is home and one is away
- exactly one team won
- exactly one team lost
- team and opponent identities cross-match
- scores mirror exactly
- run differentials are opposites
- walk-off wins pair with walk-off losses
- comeback wins pair with comeback losses
- largest leads mirror opposing largest deficits
- inning scoring splits mirror exactly

## Known 2024 distributions

- Wins: 2,428
- Losses: 2,428
- One-run wins: 675
- Wins by 2 or more: 1,753
- Wins by 4 or more: 975
- Wins by 6 or more: 479
- Shutout wins: 321
- Shutout losses: 321
- Walk-off wins: 208
- Walk-off losses: 208
- Comeback wins: 1,026
- Comeback losses: 1,026
- Team scoreless games: 321
- Team games scoring 5 or more: 2,054

These counts are regression-protected.

## Freeze rule

Phase 2B may be changed only to correct a verified factual defect.

Later memory, identity, explanation and prediction modules may consume
Phase 2B outputs but may not insert learned or subjective logic into
the team-outcome classifier.

## Next phase

Phase 2C: game anatomy and event-sequence facts.

Phase 2C will identify factual game structure such as:

- scoring innings
- decisive inning
- starter and bullpen transition points
- high-leverage plate appearances
- rallies
- stranded scoring opportunities
- bullpen entry and exit states

Explanations will be built only after those facts are verified.
