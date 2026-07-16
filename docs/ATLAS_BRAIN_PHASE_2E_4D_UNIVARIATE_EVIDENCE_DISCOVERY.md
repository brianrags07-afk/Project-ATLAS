# ATLAS Brain Phase 2E.4D — Univariate Evidence Discovery

## Status

Complete and regression-tested for the 2024 discovery season.

## Purpose

Evaluate each governed pregame feature independently against one factual
target at a time.

## Target summary

| target_name                 | grain     |   rows |   target_successes |   target_rate |   eligible_features |   tested_conditions |   strong_candidates |   candidates |   weak_candidates | prediction_weights_assigned   | predictions_created   |   elapsed_seconds | registry_path                                                                                                                                                  |
|:----------------------------|:----------|-------:|-------------------:|--------------:|--------------------:|--------------------:|--------------------:|-------------:|------------------:|:------------------------------|:----------------------|------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| target_team_win_by_2_plus   | TEAM_GAME |   4856 |               1753 |      0.360997 |                1807 |                2917 |                 121 |          175 |               175 | False                         | False                 |             17.99 | /content/drive/MyDrive/Project_Atlas/data/learning/univariate_evidence_discovery/2024/targets/target_team_win_by_2_plus/univariate_evidence_registry.parquet   |
| target_game_total_over_10   | GAME      |   2428 |                765 |      0.315074 |                3614 |                5834 |                   0 |            2 |                 0 | False                         | False                 |             43.69 | /content/drive/MyDrive/Project_Atlas/data/learning/univariate_evidence_discovery/2024/targets/target_game_total_over_10/univariate_evidence_registry.parquet   |
| target_game_total_7_or_less | GAME      |   2428 |               1058 |      0.43575  |                3614 |                5834 |                   0 |            0 |                 0 | False                         | False                 |             26.02 | /content/drive/MyDrive/Project_Atlas/data/learning/univariate_evidence_discovery/2024/targets/target_game_total_7_or_less/univariate_evidence_registry.parquet |
| target_team_win             | TEAM_GAME |   4856 |               2428 |      0.5      |                1807 |                2917 |                 116 |          159 |               161 | False                         | False                 |              8.5  | /content/drive/MyDrive/Project_Atlas/data/learning/univariate_evidence_discovery/2024/targets/target_team_win/univariate_evidence_registry.parquet             |

## Research-status summary

| target_name                 | research_status            |   conditions |
|:----------------------------|:---------------------------|-------------:|
| target_game_total_7_or_less | NOT_CONFIRMED              |         4754 |
| target_game_total_7_or_less | INSUFFICIENT_SAMPLE        |         1080 |
| target_game_total_over_10   | NOT_CONFIRMED              |         4752 |
| target_game_total_over_10   | INSUFFICIENT_SAMPLE        |         1080 |
| target_game_total_over_10   | DISCOVERY_CANDIDATE        |            2 |
| target_team_win             | NOT_CONFIRMED              |         2478 |
| target_team_win             | WEAK_DISCOVERY_CANDIDATE   |          161 |
| target_team_win             | DISCOVERY_CANDIDATE        |          159 |
| target_team_win             | STRONG_DISCOVERY_CANDIDATE |          116 |
| target_team_win             | INSUFFICIENT_SAMPLE        |            3 |
| target_team_win_by_2_plus   | NOT_CONFIRMED              |         2443 |
| target_team_win_by_2_plus   | DISCOVERY_CANDIDATE        |          175 |
| target_team_win_by_2_plus   | WEAK_DISCOVERY_CANDIDATE   |          175 |
| target_team_win_by_2_plus   | STRONG_DISCOVERY_CANDIDATE |          121 |
| target_team_win_by_2_plus   | INSUFFICIENT_SAMPLE        |            3 |

## Completion checks

| check                                | passed   |   detail |
|:-------------------------------------|:---------|---------:|
| four priority targets discovered     | True     |        4 |
| all targets produced conditions      | True     |        0 |
| duplicate condition rows zero        | True     |        0 |
| forbidden outcome fields tested zero | True     |        0 |
| all p-values valid                   | True     |        0 |
| all q-values valid                   | True     |        0 |
| prediction weights assigned false    | True     |    False |
| predictions created false            | True     |    False |

## Discovery method

For numeric features:

- lower-quartile condition,
- upper-quartile condition.

For binary features:

- high-value condition.

Each condition reports:

- available and missing samples,
- active and inactive samples,
- active and inactive success rates,
- lift,
- relative risk,
- odds ratio,
- two-proportion p-value,
- Benjamini-Hochberg q-value,
- factual effect direction.

## Governance

The research statuses are discovery labels only.

They are not:

- prediction weights,
- probabilities,
- confidence scores,
- bets,
- recommendations.

No feature combinations were tested in this phase.

No 2025 or 2026 result was used.

## Next phase

Phase 2E.4E will audit the discovered candidates for duplication, mirrored
features, availability-indicator artifacts, and identity self-reference before
forming any multifeature concepts.
