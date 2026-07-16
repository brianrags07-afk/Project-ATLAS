# ATLAS Brain Phase 2E.4F.1 — Concept Integrity Repair

## Status

Complete and regression-tested for the 2024 discovery season.

## Repair

The concept-member lineage merge now attaches only newly derived structural
lineage fields. Canonical Phase 2E.4F member feature, condition, threshold
operator, and threshold value fields remain unchanged.

This prevents pandas `_x` and `_y` column collisions.

## Purpose

Preserve all raw Phase 2E.4F concepts while preventing transformed versions of
the same underlying factual measurement from being treated as independent
baseball reasons.

## Target summary

| target_name               |   raw_concepts |   exact_joint_mask_duplicates |   inverse_joint_mask_duplicates |   same_broad_domain_concepts |   same_underlying_metric_concepts |   unique_undirected_joint_masks |   nominated_joint_mask_representatives |   nominated_independent_concepts |
|:--------------------------|---------------:|------------------------------:|--------------------------------:|-----------------------------:|----------------------------------:|--------------------------------:|---------------------------------------:|---------------------------------:|
| target_team_win           |           4176 |                             0 |                               0 |                         1808 |                                58 |                            4176 |                                   4176 |                             2368 |
| target_team_win_by_2_plus |           4388 |                             0 |                               0 |                         1710 |                                58 |                            4388 |                                   4388 |                             2678 |

## Integrity status summary

| concept_integrity_status       |   concepts |
|:-------------------------------|-----------:|
| NOMINATED_INDEPENDENT_CONCEPT  |       5046 |
| BLOCK_SINGLE_BROAD_DOMAIN      |       3402 |
| BLOCK_SHARED_UNDERLYING_METRIC |        116 |

## Completion checks

| check                                              | passed   | detail      |
|:---------------------------------------------------|:---------|:------------|
| all raw concepts preserved                         | True     | 8,564/8,564 |
| duplicate adjudicated concept IDs zero             | True     | 0           |
| duplicate nominated concept IDs zero               | True     | 0           |
| all joint masks reconstructed                      | True     | 0           |
| joint active samples match Phase 2E.4F             | True     | 0           |
| joint inactive samples match Phase 2E.4F           | True     | 0           |
| one representative per undirected joint-mask group | True     | 0           |
| nominated same-broad-domain concepts zero          | True     | 0           |
| nominated shared-underlying-metric concepts zero   | True     | 0           |
| nominated mirrored identity concepts zero          | True     | 0           |
| prediction weights assigned false                  | True     | False       |
| predictions created false                          | True     | False       |
| 2025 validation used false                         | True     | False       |
| 2026 results used false                            | True     | False       |

## Broad evidence domains

The following source classifications are consolidated into the broad domain
`IDENTITY`:

- identity edges,
- team identity summaries,
- opponent identity summaries,
- identity context.

Bullpen facts remain `BULLPEN`.

Lineup and starter facts remain `LINEUP_STARTER`.

## Feature lineage

Each concept member receives:

- its original source classification,
- its broad evidence domain,
- a feature-lineage root,
- an underlying metric root.

This allows ATLAS to recognize that team, opponent, and edge representations
of the same metric are related transformations rather than independent reasons.

## Joint activation masks

Each concept's exact joint activation pattern was reconstructed against its
original 2024 controlled discovery view.

Concepts with identical activation patterns are placed in the same exact-mask
group.

Concepts with complementary active/inactive patterns are placed in the same
undirected mask group.

## Independent nomination

A concept may enter the nominated registry only when:

- it is the selected representative of its joint-mask group,
- its members come from different broad evidence domains,
- its members do not share the same underlying metric,
- it is not a mirrored identity construction.

## Governance

- All raw Phase 2E.4F concepts are preserved.
- No raw concept was deleted.
- No prediction weight was assigned.
- No probability was created.
- No prediction was created.
- No 2025 or 2026 result was used.

## Next phase

Phase 2E.4G will freeze the nominated independent concept definitions and apply
those unchanged definitions to blind 2025 pregame evidence.
