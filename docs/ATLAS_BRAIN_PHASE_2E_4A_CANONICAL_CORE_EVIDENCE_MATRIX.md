# ATLAS Brain Phase 2E.4A — Canonical Core Pregame Evidence Matrix

## Status

Complete and regression-tested for the 2024 discovery season.

## Purpose

Create one governed team-game pregame evidence table from the factual sources
already approved in Phase 2E.

## Grain

- One row per team per game
- Join key: `game_pk + team`
- 4,856 team-game rows
- 2,428 games
- 30 teams
- Exactly two team rows per game

## Connected factual sources

| source         | path                                                                                                                                              |   rows |   columns |   unique_games |   teams |   duplicate_team_games |
|:---------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|-------:|----------:|---------------:|--------:|-----------------------:|
| identity       | /content/drive/MyDrive/Project_Atlas/data/game_intelligence/pregame_identity_matchups/2024/pregame_identity_matchups.parquet                      |   4856 |       383 |           2428 |      30 |                      0 |
| bullpen        | /content/drive/MyDrive/Project_Atlas/data/game_intelligence/clean_bullpen_pregame_facts/2024/clean_bullpen_pregame_facts.parquet                  |   4856 |        49 |           2428 |      30 |                      0 |
| lineup_starter | /content/drive/MyDrive/Project_Atlas/data/game_intelligence/clean_starter_lineup_pregame/lineup_starter_inputs/2024/lineup_starter_inputs.parquet |   4856 |       739 |           2428 |      30 |                      0 |

## Column families

| source_family        | column_role            |   columns |
|:---------------------|:-----------------------|----------:|
| bullpen              | PREGAME_FACT           |        43 |
| canonical_context    | CONTEXT                |         6 |
| identity             | PREGAME_FACT           |       377 |
| lineup_starter       | PREGAME_FACT           |       733 |
| matrix_governance    | PROVENANCE             |         1 |
| matrix_governance    | SAFETY                 |         7 |
| missingness_contract | AVAILABILITY_INDICATOR |       689 |

## Completion checks

| check                           | passed   |   detail |
|:--------------------------------|:---------|---------:|
| expected team-game rows         | True     |     4856 |
| expected unique games           | True     |     2428 |
| thirty teams represented        | True     |       30 |
| duplicate team-games zero       | True     |        0 |
| exactly two rows per game       | True     |        0 |
| identity facts included         | True     |      377 |
| bullpen facts included          | True     |       43 |
| lineup-starter facts included   | True     |      733 |
| availability indicators created | True     |      689 |
| strict backtest safe            | True     |     True |
| same-date games not used        | True     |    False |
| future games not used           | True     |    False |
| handcrafted scores excluded     | True     |    False |
| prediction values not created   | True     |    False |
| market not used                 | True     |    False |
| targets excluded                | True     |    False |
| all source joins pass           | True     |        0 |

## Governance

The matrix contains:

- prior-date team and opponent identities,
- identity differences and absolute differences,
- raw bullpen workload and availability facts,
- governed lineup-starter interaction facts,
- explicit availability indicators for fields with missing values,
- canonical game and team context.

The matrix does not contain:

- outcome targets,
- same-game results,
- handcrafted baseball scores,
- prediction probabilities,
- confidence grades,
- market information,
- future games,
- same-date completed games.

## Deferred inputs

These are not falsely marked complete:

- environment and park facts,
- weather,
- umpire,
- series/rest/travel,
- injuries and transactions,
- confirmed live-game lineup status.

They must be connected through separately governed adapters.

## Outputs

- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence.parquet`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence_column_registry.csv`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence_join_audit.csv`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence_row_audit.parquet`
- `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/canonical_pregame_evidence/2024/canonical_core_pregame_evidence_metadata.json`

## Next phase

Create the factual 2024 learning targets separately and join them only inside
the evidence-discovery workflow.

Primary target families:

- team win,
- team loss,
- win by two or more runs,
- loss by two or more runs,
- win by four or more runs,
- game total over ten,
- low-scoring game,
- high-scoring game,
- team scored five or more,
- team allowed three or fewer.

Targets remain outcomes, never pregame evidence.
