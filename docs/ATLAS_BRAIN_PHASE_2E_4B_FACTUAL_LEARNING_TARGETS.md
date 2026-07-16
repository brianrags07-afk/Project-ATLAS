# ATLAS Brain Phase 2E.4B — Factual Learning Targets

## Status

Complete and regression-tested after evidence-universe alignment.

## Discovery universe

The master completed-game source contained 2,429
2024 completed games.

The frozen Phase 2E.4A canonical pregame evidence matrix contains
2,428 games.

Only games possessing governed pregame evidence are eligible for the current
2024 discovery dataset.

No completed-game source record was deleted.

## Completed games outside the current evidence universe

|   game_pk | game_date           | home_team   | away_team   |   home_score |   away_score |   atlas_season | exclusion_reason                                                                             | source_row_deleted   | eligible_for_current_2024_discovery   |
|----------:|:--------------------|:------------|:------------|-------------:|-------------:|---------------:|:---------------------------------------------------------------------------------------------|:---------------------|:--------------------------------------|
|    746942 | 2024-06-26 00:00:00 | BOS         | TOR         |            1 |            4 |           2024 | Completed game is not present in the frozen Phase 2E.4A canonical pregame evidence universe. | False                | False                                 |

## Approved target grain

- Game targets: 2,428
- Team-game targets: 4,856
- Teams: 30
- Duplicate team-games: 0
- Games without exactly two team rows: 0

## Game summary

| metric                                                  |   value |
|:--------------------------------------------------------|--------:|
| approved discovery games                                |    2428 |
| completed games excluded from current evidence universe |       1 |
| one-run games                                           |     675 |
| games decided by 2+                                     |    1753 |
| games decided by 4+                                     |     975 |
| games total 7 or less                                   |    1058 |
| games total 10+                                         |     918 |
| games over 10 runs                                      |     765 |

## Completion checks

| check                                | passed   | detail      |
|:-------------------------------------|:---------|:------------|
| game targets match evidence universe | True     | 2,428/2,428 |
| team targets match evidence rows     | True     | 4,856/4,856 |
| thirty teams represented             | True     | 30          |
| duplicate team-games zero            | True     | 0           |
| exactly two rows per game            | True     | 0           |
| team wins equal team losses          | True     | 2,428/2,428 |
| win by 2 equals loss by 2            | True     | 1,753/1,753 |
| win by 4 equals loss by 4            | True     | 975/975     |
| all target symmetry checks pass      | True     | 0           |
| all evidence alignment checks pass   | True     | 0           |
| all universe checks pass             | True     | 0           |
| strict factual targets true          | True     | True        |
| evidence-universe aligned true       | True     | True        |
| market line not used                 | True     | False       |
| pregame evidence not embedded        | True     | False       |
| predictions not created              | True     | False       |
| future games not used                | True     | False       |

## Governance

- The master-game record remains preserved.
- Excluded games are recorded in an explicit audit artifact.
- Targets align exactly to the governed pregame evidence universe.
- Pregame evidence is not embedded in the target table.
- Market lines are not used.
- Predictions are not created.
- Future games are not used.

## Outputs

- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_team_game_learning_targets.parquet`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_game_learning_targets.parquet`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/completed_games_excluded_from_evidence_universe.csv`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_learning_target_universe_audit.csv`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_learning_target_symmetry_audit.csv`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/factual_learning_targets/2024/factual_learning_target_metadata.json`
