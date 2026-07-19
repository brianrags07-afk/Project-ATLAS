# Pregame Game Card Contract

This document defines the contract for the ATLAS **Pregame Game Card**,
whose JSON Schema lives at
[`schemas/pregame_game_card.schema.json`](../schemas/pregame_game_card.schema.json).

A Pregame Game Card is a **frozen, pregame-only snapshot** produced for a
single game. It is the only artifact that ATLAS predictions may read at
prediction time, and it must never be updated after it is frozen.

## Primary key

`card_id` (globally unique). The natural key is `(game_pk, card_version)`:
a new fact discovered after `feature_cutoff_time_utc` requires a **new**
`card_version`, never an in-place edit of an existing card.

## Sections

### card_identity
`card_id`, `game_pk`, `season`, `game_date`, `scheduled_start_time_utc`,
`home_team`, `away_team`, `venue`, `card_version`. All required.
`game_pk` (not team/player names) is the durable identifier for the game.

### temporal_provenance
`card_created_at_utc`, `feature_cutoff_time_utc`, `source_retrieved_at_utc`
(per-section retrieval timestamps), `code_commit_sha`, `pipeline_version`,
`source_manifest_id` (links to a
[pipeline manifest](PIPELINE_MANIFEST_CONTRACT.md)), `pregame_safe`
(boolean), `leakage_audit_status`. All required.

### schedule_context
`published_series_length`, `series_game_number`, `home_rest_days`,
`away_rest_days`, `home_travel_context`, `away_travel_context`,
`doubleheader_status`, and the schedule source + its timestamp. Optional
values may be `null`, but the section itself and its source/timestamp
fields are required.

### starters
`home` and `away` entries, each with: expected-or-confirmed `status`,
`confirmation_status`, `source_timestamp_utc`, `handedness`, `workload`,
`repertoire_identity`, `matchup_features`, `uncertainty_flags`.

### lineups
`home` and `away` entries, each with: `status` (expected/confirmed),
`source_timestamp_utc`, `batting_order` (position, player id, handedness),
`completeness`, `uncertainty_flags`.

### bullpen
`home` and `away` entries. `prior_usage_only` **must be `true`** -- every
bullpen fact must reflect only games completed before
`feature_cutoff_time_utc`. Includes `pitch_counts`, `appearances`, `rest`,
`likely_availability`, `roles`, `matchup_identity`, `uncertainty_flags`.

### team_and_player_memories
`team_memories` and `player_memories` blocks, each with `values`
(calculated strictly from games completed before cutoff),
`observation_count`, `recency`, `sample_sufficiency`, `version`.

### environment
`venue`, `park_factors`, `weather`, `roof_status`, `umpire`,
`source_timestamps_utc`, `missingness_flags`.

### market (optional)
Isolated from the baseball-only model. `included_in_baseball_model`
**defaults to and must be `false`** unless explicitly overridden by a
downstream consumer outside this contract. Includes `sportsbook`,
`market_type`, `price`, `observed_timestamp_utc`.

### predictions
`home_win_probability`, `away_win_probability`, `predicted_winner`,
`home_minus_1_5_probability`, `away_minus_1_5_probability`,
`projected_home_runs`, `projected_away_runs`, `projected_total`,
`over_probability`, `under_probability`, `supported_player_props`,
`fair_prices`, `confidence`, `volatility`, `uncertainty_reasons`,
`model_versions` (required).

### postgame
Always `null`/absent on a frozen pregame card. Real results are stored as
a **separate linked result object**, keyed by `card_id` and/or `game_pk`,
and must never be merged back into (mutate) the frozen pregame card.

## Required vs. optional fields

All top-level sections except `market` are required, per the JSON Schema.
Within each section, fields marked `["<type>", "null"]` in the schema are
optional-valued but the key itself must be present so downstream code can
distinguish "known absent" from "never collected."

## Flattened Parquet schema

When a Game Card corpus is materialized to Parquet for bulk analysis, each
nested object is flattened with `__` as the path separator, e.g.
`card_identity__game_pk`, `starters__home__confirmation_status`,
`predictions__home_win_probability`. Array fields (`lineups.*.batting_order`,
`predictions.supported_player_props`) are stored as JSON-encoded strings in
the flattened Parquet representation rather than exploded into rows, so
the one-row-per-card grain is preserved.

## Immutability rules

* Once `temporal_provenance.leakage_audit_status == "passed"` and the card
  is promoted out of staging, the row for a given `card_id` is
  **append-only from that point forward**: no field may be edited.
* A newly discovered pregame fact after promotion requires a new
  `card_version` (and a new `card_id`), not an update to the existing row.
* Postgame facts are never written into this object.

## Versioning rules

* `card_version` starts at `1` and increases monotonically per `game_pk`.
* `pipeline_version` and `code_commit_sha` in `temporal_provenance` must
  reflect the exact code that produced the card, matching a
  [pipeline manifest](PIPELINE_MANIFEST_CONTRACT.md) referenced by
  `source_manifest_id`.

## Cutoff rules

* `feature_cutoff_time_utc` must be less than or equal to
  `scheduled_start_time_utc`.
* Every timestamp under `source_retrieved_at_utc` must be less than or
  equal to `feature_cutoff_time_utc`. If it is not, `pregame_safe` must be
  `false` and `leakage_audit_status` must be `failed`.

## Source timestamp rules

Any field that could plausibly change between "expected" and "final"
(starters, lineups, bullpen availability, weather, market) must carry an
explicit `source_timestamp_utc` (or an entry in
`temporal_provenance.source_retrieved_at_utc`). A field with no timestamp
evidence must be treated as `pregame_possible_but_needs_timestamp_proof`,
never assumed pregame-safe.

## Expected vs. confirmed representation

`starters.*.status` and `lineups.*.status` use the enum `expected` /
`confirmed`. `confirmation_status` on starters additionally distinguishes
`unconfirmed` / `probable` / `confirmed`. Confirmed status **must** carry a
`source_timestamp_utc` at or before `feature_cutoff_time_utc`; otherwise it
must be downgraded to `expected`/`unconfirmed`.

## Leakage prevention rules

1. A completed full-season table is **not** automatically pregame-safe.
   Every value placed on a card must be traceable to a source that existed
   at or before `feature_cutoff_time_utc`.
2. No field derived from any game with a `game_date` on or after this
   card's `game_date` may be used, except the published schedule fields
   in `schedule_context` (which are pregame-safe by definition when
   sourced from the published schedule).
3. `bullpen.*.prior_usage_only` must always be `true`.
4. `postgame` must always be `null`/absent.
5. Any card failing rules 1-4 must have `pregame_safe: false` and
   `leakage_audit_status: "failed"`, and must not be promoted out of
   staging.
