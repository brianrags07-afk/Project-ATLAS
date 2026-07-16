# ATLAS Brain Phase 2E.4C — Controlled Discovery Views

## Status

Complete and regression-tested for the 2024 discovery season.

## Purpose

Create target-isolated learning datasets without modifying the canonical
pregame evidence artifact.

## Discovery grains

### Team-game grain

Used for team-perspective outcomes:

- `target_team_win`
- `target_team_win_by_2_plus`
- `target_team_win_by_4_plus`

Each dataset contains 4,856 team-game rows.

### Game grain

Used for shared game-total outcomes:

- `target_game_total_over_10`
- `target_game_total_10_plus`
- `target_game_total_7_or_less`
- `target_game_total_6_or_less`

Each dataset contains 2,428 game rows.

Home and away evidence are paired into one row. This prevents a single game
total from being counted twice.

## View registry

| target                      | grain     |   rows |   unique_games |   feature_columns |   target_successes |   target_rate | output_path                                                                                                                                   |
|:----------------------------|:----------|-------:|---------------:|------------------:|-------------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------------|
| target_team_win             | TEAM_GAME |   4856 |           2428 |              1842 |               2428 |      0.5      | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/team_game/target_team_win_discovery_view.parquet           |
| target_team_win_by_2_plus   | TEAM_GAME |   4856 |           2428 |              1842 |               1753 |      0.360997 | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/team_game/target_team_win_by_2_plus_discovery_view.parquet |
| target_team_win_by_4_plus   | TEAM_GAME |   4856 |           2428 |              1842 |                975 |      0.200783 | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/team_game/target_team_win_by_4_plus_discovery_view.parquet |
| target_game_total_over_10   | GAME      |   2428 |           2428 |              3684 |                765 |      0.315074 | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/game/target_game_total_over_10_discovery_view.parquet      |
| target_game_total_10_plus   | GAME      |   2428 |           2428 |              3684 |                918 |      0.378089 | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/game/target_game_total_10_plus_discovery_view.parquet      |
| target_game_total_7_or_less | GAME      |   2428 |           2428 |              3684 |               1058 |      0.43575  | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/game/target_game_total_7_or_less_discovery_view.parquet    |
| target_game_total_6_or_less | GAME      |   2428 |           2428 |              3684 |                785 |      0.323311 | /content/drive/MyDrive/Project_Atlas/data/learning/controlled_discovery_views/2024/game/target_game_total_6_or_less_discovery_view.parquet    |

## Completion checks

| check                         | passed   |   detail |
|:------------------------------|:---------|---------:|
| all requested views created   | True     |        7 |
| all view audits pass          | True     |        0 |
| three team-game views created | True     |        3 |
| four game-level views created | True     |        4 |
| canonical evidence unchanged  | True     |    False |
| no predictions created        | True     |    False |
| no weights assigned           | True     |    False |

## Governance

- One selected target per discovery view.
- Other target fields are excluded.
- No predictions are created.
- No weights are assigned.
- Canonical evidence is not modified.
- Market data is not used.
- Future and same-date completed games are not used.

## Next phase

Phase 2E.4D will perform univariate factual evidence discovery separately for
each target.

The first analyses will be:

1. `target_team_win_by_2_plus`
2. `target_game_total_over_10`
3. `target_game_total_7_or_less`
4. `target_team_win`

Discovery will report sample sizes, active and inactive outcome rates, lift,
effect direction, and statistical reliability. It will not assign prediction
weights.
