# ATLAS Baseball Brain Build Contract

Version: 1.0.0

## Permanent objective

ATLAS learns baseball one game at a time by connecting
pregame information with the complete game that followed.

ATLAS must learn how combinations of teams, batters,
lineups, starting pitchers, relievers, bullpens, parks,
opponents, environment, rest, travel and series context
produce:

- wins and losses
- home and away wins
- run margins
- one-run games
- wins and losses by more than 1.5, 3.5 and 5.5 runs
- shutouts and scoreless team performances
- low-scoring and high-scoring games
- overs and unders against model-defined total thresholds
- starter-driven results
- bullpen-driven results
- offensive explosions
- failed rallies
- comebacks
- blown leads
- late-game scoring
- player and lineup contributions

## Canonical season roles

- 2024: learning and identity-construction season
- 2025: unseen validation and transfer season
- 2026: production prediction and walk-forward updating

The full completed 2025 validation season may not be used
to simulate predictions within earlier 2025 dates.

## Canonical pipeline

1. Game reconstruction
2. Outcome classification
3. Game anatomy
4. Pregame-versus-actual review
5. Game explanation
6. Identity observations
7. Chronological identity updates
8. Evidence and recurring-pattern discovery
9. Prediction
10. Postgame walk-forward learning

Concepts, beliefs and evidence are downstream products.
They do not control the central architecture.

## Canonical source products

### Historical game recorder

- `data/history/game_cards/game_card_core.parquet`
- `data/history/game_cards/game_card_manifest.parquet`
- `data/history/game_cards/events/game_events_{season}_regular.parquet`

### Pregame team-game state

- `data/pregame/interactions/lineup_starter_bullpen_inputs.parquet`

### Historical starting lineups

- `data/history/lineups/historical_starting_lineups.parquet`

### Pregame bullpen state

- `data/pregame/bullpen/bullpen_pregame_state.parquet`

### Game outcomes

- `data/backtest/targets/game_targets.parquet`

### Pitcher pregame snapshots

- `data/pregame/snapshots/pitcher_pregame_snapshots.parquet`
- `data/pregame/snapshots/pitcher_game_facts.parquet`

## Canonical package policy

New Baseball Brain development belongs under:

`atlas/game_intelligence/`

The package responsibilities are:

- `contracts.py`: canonical paths, schemas and invariants
- `reconstruction.py`: read-only game reconstruction
- `outcome_classifier.py`: factual outcome labels
- `game_anatomy.py`: factual game decomposition
- `expectation_review.py`: pregame-versus-actual comparison
- `game_story.py`: structured game explanation
- `identity_observations.py`: game-level entity observations
- `walk_forward_runner.py`: chronological learning

## Legacy package policy

- `atlas/game_cards/` is the canonical historical recorder.
- `atlas/gamecards/` is legacy or daily-oriented and must not
  receive new historical brain logic.
- `atlas/identities/` is the canonical identity integration
  location for existing code.
- `atlas/identity/` currently contains no active implementation
  and must not receive new brain logic.
- `atlas/utils/` is the active utility package.
- `atlas/utilities/` must not receive new code.

No existing package or engine is deleted during the initial
brain build.

## Development rules

1. One module has one responsibility.
2. Existing working engines are reused rather than rebuilt.
3. Reconstruction remains read-only.
4. Missing information remains missing.
5. No postgame information may silently enter a pregame field.
6. Every output records source lineage and engine version.
7. Every phase must pass validation before the next phase begins.
8. Each completed phase receives a separate Git commit.
9. Earlier phases are changed only for verified defects.
10. Architecture changes require an explicit update to this file.

## Phase 1 completion criteria

The reconstruction engine must:

- reconstruct every eligible 2024 regular-season game
- return two team pregame rows per normal game
- return two lineup rows per normal game
- return two bullpen rows per normal game
- return one game-core row
- return one target row
- return the complete pitch-event sequence
- verify final scores across events, core and targets
- detect missing and duplicate keys
- verify all available pregame safety flags
- explicitly report exclusions and exceptions
- checkpoint safely
- never modify an identity or create a prediction

Known audit exception:

- `game_pk 746942`
- BOS vs TOR
- June 26, 2024
- present in game cards and lineups
- absent from interaction and target products
- must be investigated and explicitly classified
