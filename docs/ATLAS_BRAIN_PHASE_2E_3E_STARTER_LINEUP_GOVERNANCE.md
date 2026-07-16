# ATLAS Brain Phase 2E.3E — Starter and Lineup Governance

## Status

Complete and regression-tested.

## Purpose

Reuse the existing starter, pitcher, batter and lineup interaction pipeline
while removing fields that are not approved for the governed Phase 2E Brain.

## Existing pipeline confirmed

- Batter pregame snapshots: 124,457 rows
- Pitcher pregame snapshots: 52,689 rows
- Lineup-starter inputs: 12,350 rows
- Historical range: March 20, 2024 through July 3, 2026
- Regression tests: passed

## Classification repair

Historical outcome measurements now recognize singular and plural field names.

Examples preserved:

- `prior_runs_allowed`
- `rolling_strikeouts_rate`
- `career_walk_rate`
- `season_to_date_hits_allowed`
- `prior_home_run_rate`
- `rolling_innings_pitched`

These remain permitted only because the field name explicitly establishes
prior, rolling, career or season-to-date chronology.

## Governance result

- Same-date games used: no
- Future games used: no
- Handcrafted scores included: no
- Postgame outcome fields included: no
- Predictions created: no
- Existing upstream engines modified: no

## Blocked or held-for-review fields

| artifact_type        | column                                    | governance_action   | governance_reason                                          |
|:---------------------|:------------------------------------------|:--------------------|:-----------------------------------------------------------|
| LINEUP_STARTER_INPUT | uses_outcome_statistics                   | BLOCK_POSTGAME      | Explicit same-game outcome, actual result or target field. |
| LINEUP_STARTER_INPUT | uses_final_score                          | BLOCK_POSTGAME      | Explicit same-game outcome, actual result or target field. |
| LINEUP_STARTER_INPUT | live_prediction_requires_published_lineup | BLOCK_HANDCRAFTED   | Handcrafted score, grade, probability or adjustment field. |
| LINEUP_STARTER_INPUT | prediction_or_weight_assigned             | BLOCK_HANDCRAFTED   | Handcrafted score, grade, probability or adjustment field. |

## Rule

Historical pitcher and hitter results may be used only when their field names
and upstream construction establish that they were available before the
current game. Same-game results, targets and handcrafted matchup conclusions
remain excluded.
