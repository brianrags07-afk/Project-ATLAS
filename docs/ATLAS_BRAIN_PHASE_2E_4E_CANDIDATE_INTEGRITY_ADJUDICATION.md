# ATLAS Brain Phase 2E.4E — Candidate Integrity Adjudication

## Status

Complete and regression-tested for the 2024 discovery season.

## Purpose

Prevent duplicated, mirrored, complementary, or target-analogue evidence from
being counted as many independent baseball reasons.

## Target summary

| target_name               |   candidate_conditions |   unique_exact_masks |   undirected_redundancy_groups |   exact_mask_duplicate_conditions |   complement_or_inverse_conditions |   semantic_duplicate_conditions |   direct_target_analogue_conditions |   availability_indicator_conditions |   nominated_representatives |   eligible_for_multifeature_review | prediction_weights_assigned   | predictions_created   |
|:--------------------------|-----------------------:|---------------------:|-------------------------------:|----------------------------------:|-----------------------------------:|--------------------------------:|------------------------------------:|------------------------------------:|----------------------------:|-----------------------------------:|:------------------------------|:----------------------|
| target_game_total_over_10 |                      2 |                    1 |                              1 |                                 2 |                                  0 |                               2 |                                   0 |                                   0 |                           1 |                                  1 | False                         | False                 |
| target_team_win           |                    436 |                  301 |                            300 |                               236 |                                  2 |                             392 |                                  38 |                                   0 |                         300 |                                292 | False                         | False                 |
| target_team_win_by_2_plus |                    471 |                  331 |                            331 |                               246 |                                  0 |                             408 |                                  58 |                                   0 |                         331 |                                303 | False                         | False                 |

## Source-family summary

| target_name               | source_classification       |   candidate_conditions |   unique_features |   direct_target_analogues |   nominated_representatives |   eligible_for_multifeature_review |
|:--------------------------|:----------------------------|-----------------------:|------------------:|--------------------------:|----------------------------:|-----------------------------------:|
| target_game_total_over_10 | LINEUP_STARTER_PREGAME_FACT |                      2 |                 2 |                         0 |                           1 |                                  1 |
| target_team_win           | DERIVED_IDENTITY_EDGE       |                    118 |                59 |                        14 |                          92 |                                 88 |
| target_team_win           | IDENTITY_CONTEXT            |                      4 |                 2 |                         0 |                           4 |                                  4 |
| target_team_win           | LINEUP_STARTER_PREGAME_FACT |                    124 |               112 |                         0 |                          63 |                                 63 |
| target_team_win           | OPPONENT_IDENTITY_SUMMARY   |                     91 |                55 |                        12 |                          67 |                                 65 |
| target_team_win           | RAW_BULLPEN_PREGAME_FACT    |                      8 |                 6 |                         0 |                           7 |                                  7 |
| target_team_win           | TEAM_IDENTITY_SUMMARY       |                     91 |                55 |                        12 |                          67 |                                 65 |
| target_team_win_by_2_plus | DERIVED_IDENTITY_EDGE       |                    130 |                68 |                        20 |                         102 |                                 92 |
| target_team_win_by_2_plus | IDENTITY_CONTEXT            |                      4 |                 2 |                         0 |                           4 |                                  4 |
| target_team_win_by_2_plus | LINEUP_STARTER_PREGAME_FACT |                    129 |               106 |                         0 |                          67 |                                 67 |
| target_team_win_by_2_plus | OPPONENT_IDENTITY_SUMMARY   |                     91 |                60 |                        18 |                          66 |                                 58 |
| target_team_win_by_2_plus | RAW_BULLPEN_PREGAME_FACT    |                     13 |                 9 |                         0 |                          13 |                                 13 |
| target_team_win_by_2_plus | TEAM_IDENTITY_SUMMARY       |                    104 |                62 |                        20 |                          79 |                                 69 |

## Completion checks

| check                                                          | passed   | detail        |
|:---------------------------------------------------------------|:---------|:--------------|
| all discovery rows preserved                                   | True     | 17,502/17,502 |
| all candidate rows preserved                                   | True     | 909/909       |
| duplicate complete condition rows zero                         | True     | 0             |
| all candidate masks reconstructed                              | True     | 0             |
| reconstructed active samples match discovery                   | True     | 0             |
| one representative per redundancy group                        | True     | 0             |
| direct target analogues excluded from multifeature eligibility | True     | 0             |
| availability indicators excluded from multifeature eligibility | True     | 0             |
| prediction weights assigned false                              | True     | False         |
| predictions created false                                      | True     | False         |
| canonical evidence modified false                              | True     | False         |

## Exact condition masks

Each discovery condition was reconstructed against its original controlled
discovery view.

Rows were encoded as:

- 0: feature unavailable,
- 1: feature available and condition inactive,
- 2: feature available and condition active.

The resulting row pattern was hashed. Conditions with the same hash activate
on exactly the same historical rows.

A complementary hash was also created by reversing active and inactive rows
while preserving missing rows.

## Direct target analogues

Historical identities that directly restate or closely mirror the target are
preserved and flagged for review.

Examples include:

- prior win-by-two identity for the win-by-two target,
- prior won/lost identity for the win target,
- prior high-scoring identity for an over target,
- prior low-scoring identity for an under target.

These are not automatically declared leakage. They are prevented from entering
multifeature construction until separately adjudicated.

## Representative nomination

Exactly one deterministic representative is nominated per undirected
activation-mask group.

Nomination is governance only. It is not a prediction weight or statement of
baseball importance.

Direct target analogues and availability indicators are not eligible for
multifeature review.

## Governance

- All original discovery rows remain preserved.
- No discovery result was deleted.
- No prediction weights were assigned.
- No predictions were created.
- Canonical evidence was not modified.
- No 2025 or 2026 result was used.

## Next phase

Phase 2E.4F will inspect the nominated, non-analogue representatives for
cross-feature dependence and controlled multifeature concept formation.

Concepts will remain research objects until blind 2025 validation.
