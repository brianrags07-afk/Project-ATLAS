# ATLAS Brain Phase 2E.4F — Controlled Concept Formation

## Status

Complete and regression-tested for the 2024 discovery season.

## Repair applied

The original final audit compared concept feature names against one global set
of target-analogue feature names.

That produced 1,066 false positives when a feature was considered an analogue
for one target but was used for a different target.

The repaired contract validates the exact pair:

- target name,
- feature name.

True target-specific analogue violations: **0**

## Target summary

| target_name               |   pairing_members |   cross_domain_pairs_tested |   pairs_governance_eligible |   strong_concepts |   concepts |   weak_concepts |   total_candidate_concepts | prediction_weights_assigned   | predictions_created   |
|:--------------------------|------------------:|----------------------------:|----------------------------:|------------------:|-----------:|----------------:|---------------------------:|:------------------------------|:----------------------|
| target_game_total_over_10 |                 1 |                           0 |                           0 |                 0 |          0 |               0 |                          0 | False                         | False                 |
| target_team_win           |               171 |                       11388 |                        5365 |              2631 |       1124 |             421 |                       4176 | False                         | False                 |
| target_team_win_by_2_plus |               177 |                       12372 |                        6000 |              2653 |       1191 |             544 |                       4388 | False                         | False                 |

## Domain-pair summary

| target_name               | domain_pair                                             |   pairs_tested |   governance_eligible_pairs |   strong_concepts |   concepts |   weak_concepts |   median_jaccard |   median_absolute_phi |   median_incremental_lift |
|:--------------------------|:--------------------------------------------------------|---------------:|----------------------------:|------------------:|-----------:|----------------:|-----------------:|----------------------:|--------------------------:|
| target_team_win           | DERIVED_IDENTITY_EDGE + IDENTITY_CONTEXT                |            160 |                          80 |                26 |         14 |              16 |         0.12399  |             0.219744  |               0.0115001   |
| target_team_win           | DERIVED_IDENTITY_EDGE + LINEUP_STARTER_PREGAME_FACT     |           1600 |                         800 |               339 |        151 |              82 |         0.150922 |             0.0857063 |               0.00330247  |
| target_team_win           | DERIVED_IDENTITY_EDGE + OPPONENT_IDENTITY_SUMMARY       |           1600 |                         800 |               374 |        186 |              38 |         0.17396  |             0.277148  |               0.0219683   |
| target_team_win           | DERIVED_IDENTITY_EDGE + RAW_BULLPEN_PREGAME_FACT        |            280 |                         140 |                74 |         32 |              11 |         0.208561 |             0.175625  |               0.0177034   |
| target_team_win           | DERIVED_IDENTITY_EDGE + TEAM_IDENTITY_SUMMARY           |           1600 |                         800 |               374 |        186 |              38 |         0.17396  |             0.277148  |               0.0219683   |
| target_team_win           | IDENTITY_CONTEXT + LINEUP_STARTER_PREGAME_FACT          |            160 |                          80 |                43 |         12 |               7 |         0.145536 |             0.0556017 |              -0.000966271 |
| target_team_win           | IDENTITY_CONTEXT + OPPONENT_IDENTITY_SUMMARY            |            160 |                          80 |                48 |         19 |              13 |         0.148919 |             0.154533  |               0.016223    |
| target_team_win           | IDENTITY_CONTEXT + RAW_BULLPEN_PREGAME_FACT             |             28 |                          14 |                11 |          0 |               0 |         0.151262 |             0.0346181 |               0.0130745   |
| target_team_win           | IDENTITY_CONTEXT + TEAM_IDENTITY_SUMMARY                |            160 |                          80 |                48 |         19 |              13 |         0.148919 |             0.154533  |               0.016223    |
| target_team_win           | LINEUP_STARTER_PREGAME_FACT + OPPONENT_IDENTITY_SUMMARY |           1600 |                         722 |               482 |         97 |              41 |         0.147557 |             0.0349166 |              -0.0285326   |
| target_team_win           | LINEUP_STARTER_PREGAME_FACT + RAW_BULLPEN_PREGAME_FACT  |            280 |                         149 |                58 |         38 |              16 |         0.164307 |             0.0712609 |               0.00966864  |
| target_team_win           | LINEUP_STARTER_PREGAME_FACT + TEAM_IDENTITY_SUMMARY     |           1600 |                         878 |               393 |        157 |              85 |         0.153755 |             0.117616  |               0.00588505  |
| target_team_win           | OPPONENT_IDENTITY_SUMMARY + RAW_BULLPEN_PREGAME_FACT    |            280 |                         101 |                51 |         37 |               9 |         0.157668 |             0.0270869 |              -0.0198986   |
| target_team_win           | OPPONENT_IDENTITY_SUMMARY + TEAM_IDENTITY_SUMMARY       |           1600 |                         462 |               220 |        138 |              38 |         0.140898 |             0.0267406 |              -0.0467452   |
| target_team_win           | RAW_BULLPEN_PREGAME_FACT + TEAM_IDENTITY_SUMMARY        |            280 |                         179 |                90 |         38 |              14 |         0.268478 |             0.250277  |               0.0166488   |
| target_team_win_by_2_plus | DERIVED_IDENTITY_EDGE + IDENTITY_CONTEXT                |            160 |                          80 |                 8 |          4 |               8 |         0.12399  |             0.234335  |               0.00526488  |
| target_team_win_by_2_plus | DERIVED_IDENTITY_EDGE + LINEUP_STARTER_PREGAME_FACT     |           1600 |                         796 |               337 |        152 |              76 |         0.161376 |             0.103499  |               0.00544419  |
| target_team_win_by_2_plus | DERIVED_IDENTITY_EDGE + OPPONENT_IDENTITY_SUMMARY       |           1600 |                         824 |               354 |        184 |              57 |         0.23525  |             0.266998  |               0.0193666   |
| target_team_win_by_2_plus | DERIVED_IDENTITY_EDGE + RAW_BULLPEN_PREGAME_FACT        |            520 |                         259 |               107 |         37 |              29 |         0.141908 |             0.137645  |               0.000670504 |
| target_team_win_by_2_plus | DERIVED_IDENTITY_EDGE + TEAM_IDENTITY_SUMMARY           |           1600 |                         786 |               216 |        161 |              91 |         0.101487 |             0.268159  |               0.00643445  |
| target_team_win_by_2_plus | IDENTITY_CONTEXT + LINEUP_STARTER_PREGAME_FACT          |            160 |                          80 |                32 |         27 |               7 |         0.14549  |             0.0633617 |               0.00437439  |
| target_team_win_by_2_plus | IDENTITY_CONTEXT + OPPONENT_IDENTITY_SUMMARY            |            160 |                          80 |                26 |         23 |              10 |         0.144741 |             0.139111  |               0.00836603  |
| target_team_win_by_2_plus | IDENTITY_CONTEXT + RAW_BULLPEN_PREGAME_FACT             |             52 |                          26 |                 9 |          8 |               1 |         0.147954 |             0.0313473 |               0.00254101  |
| target_team_win_by_2_plus | IDENTITY_CONTEXT + TEAM_IDENTITY_SUMMARY                |            160 |                          80 |                12 |          7 |              19 |         0.150476 |             0.171856  |              -0.00689878  |
| target_team_win_by_2_plus | LINEUP_STARTER_PREGAME_FACT + OPPONENT_IDENTITY_SUMMARY |           1600 |                         752 |               507 |        120 |              47 |         0.151642 |             0.0292539 |              -0.0218428   |
| target_team_win_by_2_plus | LINEUP_STARTER_PREGAME_FACT + RAW_BULLPEN_PREGAME_FACT  |            520 |                         262 |               106 |         63 |              26 |         0.15013  |             0.055116  |               0.000307764 |
| target_team_win_by_2_plus | LINEUP_STARTER_PREGAME_FACT + TEAM_IDENTITY_SUMMARY     |           1600 |                         828 |               344 |        170 |              85 |         0.163165 |             0.125755  |              -0.000607703 |
| target_team_win_by_2_plus | OPPONENT_IDENTITY_SUMMARY + RAW_BULLPEN_PREGAME_FACT    |            520 |                         248 |               141 |         40 |              17 |         0.151515 |             0.0270029 |              -0.0129118   |
| target_team_win_by_2_plus | OPPONENT_IDENTITY_SUMMARY + TEAM_IDENTITY_SUMMARY       |           1600 |                         632 |               345 |        137 |              48 |         0.146284 |             0.0295751 |              -0.0482267   |
| target_team_win_by_2_plus | RAW_BULLPEN_PREGAME_FACT + TEAM_IDENTITY_SUMMARY        |            520 |                         267 |               109 |         58 |              23 |         0.161515 |             0.173847  |               0.00453664  |

## Completion checks

| check                                        | passed   |   detail |
|:---------------------------------------------|:---------|---------:|
| cross-domain pairs created                   | True     |    23760 |
| same-domain pair failures zero               | True     |        0 |
| duplicate concept IDs zero                   | True     |        0 |
| exactly two members per concept              | True     |        0 |
| target-specific direct analogue members zero | True     |        0 |
| all concepts have sufficient joint samples   | True     |        0 |
| all concepts pass overlap controls           | True     |        0 |
| all concepts add incremental value           | True     |        0 |
| prediction weights assigned false            | True     |    False |
| predictions created false                    | True     |    False |
| canonical evidence modified false            | True     |    False |

## Pair controls

Members must:

- belong to the same factual target,
- come from different governed source domains,
- have compatible effect directions,
- have sufficient historical samples,
- remain below the Jaccard-overlap ceiling,
- remain below the absolute-phi dependence ceiling.

## Concept controls

The joint condition must:

- activate at least 25 rows,
- leave at least 25 inactive rows,
- add at least 0.015 outcome-rate lift beyond the stronger member,
- pass target-specific multiple-testing correction.

## Governance

- Direct target analogues are checked per target.
- Availability artifacts remain excluded.
- Same-domain pairs are not formed.
- No prediction weights are assigned.
- No probabilities are created.
- No predictions are created.
- No 2025 or 2026 result is used.

## Next phase

Phase 2E.4G freezes these 2024 concept definitions and evaluates their exact
activation rules blindly against 2025.
