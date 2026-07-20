# ATLAS Baseball Brain Dependency Graph (2024 Learning Season)

This document builds the dependency graph requested for reconstructing
the 2024 Baseball Brain. Per the task's explicit new requirement, it
contains **two kinds of nodes, not one**:

- **Software dependencies** — existing tables, engines, pipeline stages,
  and contracts already present in this repository (`atlas/`,
  `atlas_reference/`, `schemas/`).
- **Baseball-knowledge dependencies** — the historical baseball facts
  ATLAS must learn to explain *why* games unfold the way they do, per
  the project's stated objective. Every one of these is mapped to a
  software dependency if one already exists, or explicitly flagged as
  a **future canonical dataset** if it does not — even if that dataset
  has never existed before in this project.

Nothing in this document authorizes a rebuild by itself. It is a map to
be reviewed before any rebuild work begins, per
`docs/ATLAS_DATA_ALIGNMENT_AUDIT.md`.

## 1. Anchor artifacts

The graph is anchored on the two contracts every Game Card / rebuild run
must ultimately satisfy:

- `docs/PREGAME_GAME_CARD_CONTRACT.md` / `schemas/pregame_game_card.schema.json`
  — the frozen, pregame-only snapshot: `card_identity`,
  `temporal_provenance`, `schedule_context`, `starters`, `lineups`,
  `bullpen`, `team_and_player_memories`, `environment`, `market`
  (optional), `predictions`, `postgame` (always null on this object).
- `docs/PIPELINE_MANIFEST_CONTRACT.md` / `schemas/pipeline_manifest.schema.json`
  — the append-only run record every builder must produce before
  promotion out of staging.

**Gap identified here, not previously documented:** the Pregame Game
Card contract explicitly stores real results as "a separate linked
result object" and states `postgame` "must always be null/absent" on
the pregame card. Neither this repository's schemas nor its docs
currently define **what that separate postgame/historical object looks
like**. The task's five required Game Card sections — (1) pregame
state, (2) complete game anatomy, (3) structured Game Story explaining
WHY, (4) learning observations, (5) identity updates — map as follows:

| Required section | Current status |
|---|---|
| 1. Pregame state | **Covered** by the existing Pregame Game Card contract above. |
| 2. Complete game anatomy | **Partially covered.** `atlas/game_intelligence/game_flow_fact_table.py`, `scoring_state_timeline.py`, `scoring_event_roles.py`, `team_game_flow.py` compute game-flow facts, but there is no single contract/schema that assembles them into one "game anatomy" record per game. |
| 3. Structured Game Story (WHY) | **Not covered — future canonical artifact.** No schema, contract, or table anywhere in the repo produces a structured narrative/causal explanation of a game. Supporting engines exist (`outcome_classifier.py`, `team_outcome_classifier.py`, `response_recovery.py`, `lead_protection.py`) but their outputs are classifications/facts, not an assembled "story." |
| 4. Learning observations | **Partially covered.** `atlas/learning/evidence_consolidation_engine.py`, `team_evidence_discovery.py`, `league_evidence_discovery.py`, `univariate_evidence_discovery.py` produce evidence/observations, but there is no per-game "learning observations" record linked back to a `game_pk`/Game Card. |
| 5. Identity updates | **Partially covered.** `atlas/game_intelligence/pregame_team_identity_timeline.py` and `atlas/identities/bullpen_identity_integration_engine.py` update identities *before* a game (consumed by the pregame card); there is no equivalent "what did this specific completed game teach us about identity X" record after the fact. |

**Recommendation:** define a new, explicitly historical/postgame
contract (working name: `historical_game_card` or `game_story_record`,
keyed by `game_pk` + a monotonic `story_version`, mirroring the Pregame
Game Card's append-only/versioning rules) that assembles sections 2–5.
This is new-artifact design work, not a rebuild of anything that
exists — flagged for review, not started here.

## 2. Software dependency chain for the 2024 Pregame Game Card (what exists today)

```
master_game_database.parquet ─┬─> game_intelligence/pregame_team_identity_timeline.py ─> pregame_team_identities/2024/*.parquet ─┐
master_pitch_database.parquet ┘                                                                                                    │
                                                                                                                                     ├─> game_intelligence/pregame_identity_matchup_builder.py ─> pregame_identity_matchups/2024/*.parquet ─┐
identities/bullpen_availability_fatigue_engine.py <── master_game_database.parquet, master_pitch_database.parquet                  │                                                                                                        │
        │                                                                                                                          │                                                                                                        │
        └─> identities/bullpen_identity_integration_engine.py ─> pregame/clean_bullpen_pregame_facts.py ─> clean_bullpen_pregame_facts/2024/*.parquet ─────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                                                                                                                                              │
pregame/clean_starter_lineup_pregame.py <── master_game_database.parquet, master_pitch_database.parquet ─> batter_pregame_snapshots/2024/*.parquet ────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                                                                                                                                              │
memory/memory_engine.py <── team/player evidence (atlas/evidence/) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                                                                                                                                              ▼
                                                                                                                       pregame/canonical_core_evidence_matrix.py ──> Pregame Game Card (schemas/pregame_game_card.schema.json)
```

This chain is a **software-only** view, reconstructed from module names
and imports. It is corroborated, for the 2025-side of the pipeline, by a
more authoritative, code-verified source: `atlas_reference/
dev_data_bundle_required_artifacts.json` already lists 13 required
production artifacts with `primary_key` and `consuming_builder` fields
recorded directly from the real Colab/Drive production tree, including
`clean_bullpen_pregame_facts_2025`, `batter_pregame_snapshots_2025`,
`pitcher_pregame_snapshots_2025`, `historical_starting_lineups`, and
`game_anomaly_registry` (consumed by
`atlas/interactions/lineup_starter_input_engine.py`). **This document's
2024 chain should be re-derived directly from that registry's pattern
(and extended to a `..._2024` allowlist) rather than solely from static
analysis**, once a maintainer can regenerate it for 2024.

Every arrow above depends on `master_game_database.parquet` /
`master_pitch_database.parquet` being canonical for 2024 — which, per
`docs/ATLAS_DATA_ALIGNMENT_AUDIT.md` Section 2, is **currently
unverified** (Drive vs. GCS bucket ambiguity, though the dev-data-bundle
registry is the best current lead on the Drive side). The chain is
directionally correct per the code that exists, but every node inherits
that upstream uncertainty until resolved.

## 3. Baseball-knowledge dependency map

For each baseball-learning objective named in the task, the table below
records: the existing question(s) it maps to in
`atlas/questions/question_library.py` (already encodes intended WHEN-context
tags), the existing engine(s)/table(s) that back it today, and whether a
canonical dataset actually exists.

| Baseball-knowledge objective | Existing question-library context tag(s) | Existing engine(s) | Canonical dataset today | Status |
|---|---|---|---|---|
| Player strengths/weaknesses | `starter_identity`, `offense_identity`, `opponent_identity` | `atlas/evidence/`, `atlas/profiles/`, `atlas/learning/team_evidence_discovery.py` | Partial (evidence tables, no single player-strength dataset) | **Existing, incomplete** |
| Team strengths/weaknesses | `offense_identity`, `opponent_identity`, `team_margin` | `atlas/teams/`, `atlas/evidence/team_evidence.py` | Partial | **Existing, incomplete** |
| Hitter identities | (implicit in `offense_identity`) | `atlas/pregame/clean_starter_lineup_pregame.py` (`batter_pregame_snapshots`) | Present for 2024/2025/2026 per `table_catalog.json` | **Existing** (trust unverified per Section 2) |
| Starter identities | `starter_identity`, `starter_advantage`, `starter_matchup` | `atlas/pitchers/`, `atlas/game_intelligence/pregame_identity_matchup_builder.py` | `pregame_identity_matchups/2024` | **Existing** (trust unverified) |
| Bullpen identities | `bullpen_state`, `bullpen_advantage` | `atlas/identities/bullpen_identity_integration_engine.py`, `atlas/identities/bullpen_availability_fatigue_engine.py` | `clean_bullpen_pregame_facts/2024,2025,2026` | **Existing** (trust unverified) |
| Lineup identities | (implicit; `lineups.csv` daily files) | `atlas/pregame/clean_starter_lineup_pregame.py` | `data/daily/*/lineups.csv`, `batter_pregame_snapshots` | **Existing, partial** — no dedicated lineup-*identity* (vs. per-batter snapshot) table found in the catalog. |
| Park effects | `park` | none found under `atlas/` | **Not in `COVERAGE_ROWS`; not in `table_catalog.json`** | **Future canonical dataset required**: `park_factors` (distinct from `venue`, which only names the venue, not its run/HR/handedness effects). |
| Weather | `weather` | none found under `atlas/`; row exists in `coverage_matrix.COVERAGE_ROWS` | Zero matches for "weather" in `table_catalog.json` | **Tracked by the audit tool, but no data or builder exists.** Confirmed gap. |
| Umpire tendencies | `umpire` | none found under `atlas/`; row exists in `coverage_matrix.COVERAGE_ROWS` | Zero matches for "umpire" in `table_catalog.json` | **Tracked by the audit tool, but no data or builder exists.** Confirmed gap — and per the audit's own published-schedule rule design (Section on leakage in `ATLAS_HISTORICAL_READINESS_AUDIT.md`), umpire assignment must be evaluated for `pregame_safety` carefully: umpire-per-game is often not published/confirmed until close to game time. |
| Rest | `rest` | row exists in `coverage_matrix.COVERAGE_ROWS`, `DYNAMIC_PREGAME_ROWS` | Not found as a standalone dataset; likely derivable from `master_game_database` schedule gaps | **Existing in principle, not materialized** — needs a builder, not new raw data. |
| Travel | `travel` | row exists in `coverage_matrix.COVERAGE_ROWS`, `DYNAMIC_PREGAME_ROWS` | Not found; requires venue-to-venue distance/timezone reference data not present anywhere in `atlas_reference/` | **Future canonical dataset required**: `travel_context` (venue geocoordinates/timezone reference + derived travel distance/direction per game). |
| Series context | `series_game`, `published_series_context` | `atlas/audit/schedule_source_assessment.py` already implements the no-leakage rule for this | Governed by the same leakage rule as `published_schedule`; data presence unverified | **Existing rule, data unverified.** |
| Pitch-type interactions | `pitch_type_profile` | `atlas/pitchers/v2/pitch_table.py` | `master_pitch_database.parquet` (pitch-level) exists per contract; pitch **arsenal/interaction** features not confirmed as materialized | **Existing raw data, missing derived feature dataset.** |
| Handedness interactions | (implicit in matchup features; not an explicit tag) | `starters.*.handedness`, `lineups.*.batting_order[].handedness` fields exist in the Pregame Game Card schema | No standalone handedness-splits dataset found in `table_catalog.json` | **Future canonical dataset recommended**: `handedness_splits` (batter-vs-hand and pitcher-vs-hand rate tables), even though the raw at-bat data to derive it likely already exists in `master_pitch_database.parquet`. |
| Game flow | `game_flow` (focus-area, not yet a question-library tag) | `atlas/game_intelligence/game_flow_fact_table.py`, `scoring_state_timeline.py`, `lead_protection.py`, `response_recovery.py` | Present as engines; no single "game flow" output dataset confirmed | **Existing engines, output dataset unconfirmed.** |
| Manager tendencies | none found | none found | Zero matches for "manager" in `table_catalog.json` | **Future canonical dataset required**: `manager_tendencies` (bullpen-usage patterns, pinch-hit/steal aggressiveness, lineup-construction habits, by manager and by game situation). Never existed in this project. |
| Roster / transaction movement | none found | none found | Zero matches for "roster" beyond one 2025 validation input-readiness column-inventory file; zero matches for "transaction" | **Future canonical dataset required**: `roster_transactions` (call-ups, IL moves, trades, DFA — with transaction timestamps, since these directly gate pregame-safety of "who is even on this roster today"). |
| "Why games unfold the way they do" (Game Story) | not a single tag; spans `game_flow`, `starter_matchup`, `bullpen_advantage`, etc. | `outcome_classifier.py`, `team_outcome_classifier.py` | No structured narrative output exists | **Future canonical dataset required** — see Section 1 (`historical_game_card` / `game_story_record`). |
| Learning observations (per game) | n/a | `atlas/learning/evidence_consolidation_engine.py` and the `*_evidence_discovery.py` family | Evidence exists at season/concept grain, not confirmed at per-game grain | **Future canonical dataset required**: a per-game `learning_observations` record linking a completed game to which concept-evidence rows it contributed to. |
| Identity updates (per game) | n/a | `pregame_team_identity_timeline.py` produces the *pregame* snapshot; no *postgame* "this game updated identity X because Y" record found | Not found | **Future canonical dataset required**: `identity_update_log` (per-game, per-identity delta with the specific evidence that caused it). |

## 4. Unified graph: software nodes + baseball-knowledge nodes

Reading order: a baseball-knowledge objective (left) is either satisfied
by an existing software chain (middle) or requires a future canonical
dataset (right) before the historical Game Card's five sections can be
fully populated.

```
[Player/team strengths & weaknesses] --partial--> atlas/evidence/*, atlas/profiles/* --> team_and_player_memories (Pregame Game Card)
[Hitter identities]                  --exists---> clean_starter_lineup_pregame.py --> batter_pregame_snapshots/2024 --> lineups (Pregame Game Card)
[Starter identities]                 --exists---> pregame_identity_matchup_builder.py --> pregame_identity_matchups/2024 --> starters (Pregame Game Card)
[Bullpen identities]                 --exists---> bullpen_identity_integration_engine.py --> clean_bullpen_pregame_facts/2024 --> bullpen (Pregame Game Card)
[Lineup identities]                  --partial--> clean_starter_lineup_pregame.py --> lineups (Pregame Game Card)
[Park effects]                       --MISSING--> [FUTURE: park_factors dataset] --> environment.park_factors (Pregame Game Card)
[Weather]                            --MISSING--> [FUTURE: weather dataset]      --> environment.weather (Pregame Game Card)
[Umpire tendencies]                  --MISSING--> [FUTURE: umpire_tendencies dataset] --> environment.umpire (Pregame Game Card)
[Rest]                                --partial--> [needs builder over master_game_database] --> schedule_context.*_rest_days (Pregame Game Card)
[Travel]                             --MISSING--> [FUTURE: travel_context dataset] --> schedule_context.*_travel_context (Pregame Game Card)
[Series context]                     --exists (rule)--> schedule_source_assessment.py --> schedule_context.published_series_length (Pregame Game Card)
[Pitch-type interactions]            --partial--> pitch_table.py over master_pitch_database.parquet --> starters.*.repertoire_identity, matchup_features (Pregame Game Card)
[Handedness interactions]            --MISSING--> [FUTURE: handedness_splits dataset] --> starters.*.matchup_features, lineups.*.batting_order[].handedness (Pregame Game Card)
[Game flow]                          --partial--> game_flow_fact_table.py, scoring_state_timeline.py, lead_protection.py, response_recovery.py --> [FUTURE: game_anatomy section]
[Manager tendencies]                 --MISSING--> [FUTURE: manager_tendencies dataset] --> [no current Game Card section — would extend team_and_player_memories or a new section]
[Roster / transaction movement]      --MISSING--> [FUTURE: roster_transactions dataset] --> gates pregame-safety of starters/lineups/bullpen sections
[Game Story: WHY]                    --MISSING--> outcome_classifier.py, team_outcome_classifier.py (facts only) --> [FUTURE: game_story_record]
[Learning observations]              --partial--> evidence_consolidation_engine.py, *_evidence_discovery.py --> [FUTURE: per-game learning_observations record]
[Identity updates]                   --partial--> pregame_team_identity_timeline.py (pregame direction only) --> [FUTURE: identity_update_log]
```

## 5. Reading the graph: what blocks a trustworthy 2024 rebuild today

1. **Everything is downstream of the Drive-vs-GCS-bucket resolution**
   (`ATLAS_DATA_ALIGNMENT_AUDIT.md` Section 2). No node above can be
   promoted from "exists" to "canonical" until that is settled.
2. **Seven genuinely new canonical datasets are required, not rebuilt**:
   `park_factors`, `weather`, `umpire_tendencies`, `travel_context`,
   `handedness_splits`, `manager_tendencies`, `roster_transactions` —
   none of which have ever existed in this project. These should go through the same schema-first process
   as everything else in `atlas_reference/schemas/` (define the schema
   and sample before any builder code is written), consistent with the
   "never invent column names" / "check registries first" governance
   already in force for ATLAS data builders.
3. **Four future artifacts assemble the historical/postgame half of the
   Game Card** (`game_story_record` covering game anatomy, Game Story,
   learning observations, and identity updates) — these are net-new
   contract/schema work, analogous to how
   `PREGAME_GAME_CARD_CONTRACT.md` was defined, not analogous to any
   existing table.
4. **`rest` and pitch-type interaction features are the cheapest wins**:
   the raw data likely already exists (`master_game_database` schedule
   gaps for rest; `master_pitch_database` for pitch-type), and only a
   builder is missing — no new upload is required for these two.

## 6. What this graph does not do

- It does not authorize uploading anything. See
  `docs/ATLAS_2024_BRAIN_UPLOAD_MANIFEST.md` for scoped upload
  recommendations derived from this graph.
- It does not modify `atlas/audit/coverage_matrix.py`'s `COVERAGE_ROWS`.
  Adding `park_factors`, `handedness_splits`, `manager_tendencies`,
  `roster_transactions`, `game_story`, `game_anatomy`,
  `learning_observations`, and `identity_updates` as new tracked rows is
  a recommended follow-up code change, not performed in this
  documentation-only session.
