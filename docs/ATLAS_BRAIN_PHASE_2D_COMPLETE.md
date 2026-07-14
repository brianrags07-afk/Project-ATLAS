# ATLAS Brain Phase 2D Complete

## Status

Phase 2D is frozen and complete.

Phase 2D establishes the verified factual game-flow language used
to describe how each 2024 MLB game developed after scoring began.

## Frozen Engines

- `atlas/game_intelligence/scoring_event_roles.py`
- `atlas/game_intelligence/scoring_event_role_season_builder.py`
- `atlas/game_intelligence/team_game_flow.py`
- `atlas/game_intelligence/lead_protection.py`
- `atlas/game_intelligence/response_recovery.py`
- `atlas/game_intelligence/game_flow_fact_table.py`

## Frozen Artifacts

### Scoring-event roles

- `data/game_intelligence/scoring_event_roles/2024/scoring_event_roles.parquet`
- `data/game_intelligence/scoring_event_roles/2024/scoring_event_role_audit.parquet`
- `data/game_intelligence/scoring_event_roles/2024/scoring_event_role_failures.parquet`
- `data/game_intelligence/scoring_event_roles/2024/scoring_event_role_metadata.json`

### Team game flow

- `data/game_intelligence/team_game_flow/2024/team_game_flow.parquet`
- `data/game_intelligence/team_game_flow/2024/team_game_flow_audit.parquet`

### Lead protection and separation

- `data/game_intelligence/lead_protection/2024/team_lead_protection.parquet`
- `data/game_intelligence/lead_protection/2024/team_lead_protection_audit.parquet`

### Response and recovery

- `data/game_intelligence/response_recovery/2024/team_response_recovery.parquet`
- `data/game_intelligence/response_recovery/2024/team_response_recovery_audit.parquet`

### Consolidated fact table

- `data/game_intelligence/game_flow_facts/2024/team_game_flow_facts.parquet`
- `data/game_intelligence/game_flow_facts/2024/team_game_flow_fact_audit.parquet`
- `data/game_intelligence/game_flow_facts/2024/team_game_flow_fact_failures.parquet`
- `data/game_intelligence/game_flow_facts/2024/team_game_flow_fact_metadata.json`

## Completion Metrics

- Verified games: 2,428
- Scoring-event role rows: 16,450
- Team-game rows: 4,856
- Consolidated columns: 121
- Teams represented: 30
- Opening scores: 2,428
- Decisive scoring events: 2,428
- One-run winners: 675
- Winners by two or more runs: 1,753
- Winners by three or more runs: 1,316
- Teams reaching a two-run lead: 2,591
- Teams reaching two runs of separation but failing -1.5: 838
- Teams leading by two or more but losing: 491
- Wins after allowing the first score: 765
- Same-inning responses: 2,269
- Cross-layer failures: 0
- Audit failures: 0
- Saved failure rows: 0
- Regression tests passed: 23

## Factual Scoring Roles Frozen

Every scoring transition has exactly one primary role:

- Opening score
- Tying score
- Go-ahead score
- Lead extension
- Deficit reduction

Every game contains exactly one decisive scoring event.

## Run-Line Foundation Frozen

Phase 2D explicitly records:

- Winner by exactly one run
- Winner by two or more runs
- Winner by three or more runs
- Historical -1.5 coverage
- Historical +1.5 coverage
- Whether a team reached a two-run lead
- Whether a two-run lead survived to the final
- Whether a team reached two-run separation but failed -1.5
- Whether a team led by two or more and lost
- Whether a winner gave back part of its maximum lead
- Whether a winner failed to separate

These are postgame factual targets and are not directly pregame-safe.

Future pregame models may consume only historical or lagged
aggregates created strictly from games completed before the target
game.

## Response and Recovery Foundation Frozen

Phase 2D records:

- Scored first
- Allowed first score
- Won after allowing first score
- Lost after scoring first
- Immediate response
- Eventual response
- Same-inning response
- Response within one inning
- Tying response
- Go-ahead response
- Late response
- Longest unanswered opponent scoring streak

## Provenance Rules Frozen

1. Phase 2D is postgame factual data.
2. Phase 2D rows are not directly pregame-feature safe.
3. No sportsbook information is used.
4. No predictions are created.
5. No identities are modified.
6. No explanations are created.
7. No future games are used.
8. Phase 2B remains frozen and unchanged.
9. Phase 2B compatibility aliases are derived only inside the
   Phase 2D consolidated table.
10. Every shared score, margin, opponent, win/loss and run-line
    field must agree across layers.

## Regression Test File

- `tests/test_game_flow_intelligence.py`

## Freeze Rule

Future phases may consume Phase 2D artifacts but may not silently
alter Phase 2D definitions or historical outputs.

Any Phase 2D change requires:

1. A documented defect.
2. A reproducible failing game or invariant.
3. Full 2024 reconstruction and regression testing.
4. An explicit version change when definitions change.
5. A new milestone commit.
