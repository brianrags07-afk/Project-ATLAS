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

## 1a. Explicit run-margin / run-line learning-objective lineage

ATLAS's stated learning objective includes an explicit run-margin/
run-line objective, not only a win/loss objective. This section traces
that lineage end to end, node by node, using only column/module names
that already exist in the repository.

```
master_game_database.parquet (canonical final scores: home_score, away_score)
  └─> atlas/game_intelligence/outcome_classifier.py
        (per-game factual outcome classifier: computes
         absolute_run_margin, one_run_game, plus the
         won_by_{2,4,6}_plus / lost_by_{2,4,6}_plus family
         via team_outcome_classifier.py below)
        │
        └─> atlas/game_intelligence/team_outcome_classifier.py
              (team-game grain: run_differential, absolute_run_margin,
               one_run_game, won_by_2_plus, won_by_4_plus, won_by_6_plus,
               lost_by_2_plus, lost_by_4_plus, lost_by_6_plus,
               covered_minus_1_5_result, covered_minus_3_5_result,
               covered_minus_5_5_result — team-game outcome facts)
              │
              └─> atlas/learning/backtest_target_builder.py,
                  atlas/learning/factual_target_builder.py
                  (team-game targets: build_game_targets /
                   build_team_game_targets read `won` and
                   `run_differential` into the canonical
                   `team_game_targets.parquet` artifact)
                    │
                    └─> atlas/validation/target_resolution.py
                        (materializes the frozen target name
                         `target_team_win_by_2_plus` — see
                         `TARGET_TEAM_WIN_BY_2_PLUS` — from the
                         canonical `won` / `run_differential` columns,
                         under `FROZEN_TARGET_RESOLUTION_RULES`)
                          │
                          ├─> 2024 learning: `target_team_win_by_2_plus`
                          │   is a named target in
                          │   `atlas/learning/candidate_integrity_adjudication.py`
                          │   and in the univariate evidence-discovery
                          │   schemas
                          │   (`.../univariate_evidence_discovery/2024/
                          │   targets/target_team_win_by_2_plus/...`)
                          │     │
                          │     └─> frozen concept definitions (2024)
                          │         — `docs/ATLAS_BRAIN_PHASE_2E_4G_IMMUTABLE_CONCEPT_FREEZE.md`
                          │           (2,138 concepts / 4,276 members,
                          │           hash-fingerprinted; any concept
                          │           built on `target_team_win_by_2_plus`
                          │           is frozen alongside every other
                          │           2024 concept)
                          │             │
                          │             └─> 2025 blind validation —
                          │                 `atlas/validation/
                          │                 concept_validation_2025.py`
                          │                 calls
                          │                 `resolve_frozen_targets`
                          │                 (from `target_resolution.py`)
                          │                 and certifies rule
                          │                 consistency via
                          │                 `certify_target_resolution_matches_rules`
                          │                 before scoring any frozen
                          │                 concept, including ones keyed
                          │                 on `target_team_win_by_2_plus`
                          │                   │
                          │                   └─> [FUTURE: dedicated
                          │                       run-line prediction
                          │                       model] — no module
                          │                       under `atlas/` currently
                          │                       trains/serves a
                          │                       standalone run-line
                          │                       (spread) prediction
                          │                       model; only the
                          │                       factual target and its
                          │                       2024/2025 concept
                          │                       lineage exist today
                          │                         │
                          │                         └─> Pregame Game Card
                          │                             `predictions.
                          │                             home_minus_1_5_probability`,
                          │                             `predictions.
                          │                             away_minus_1_5_probability`
                          │                             (`schemas/
                          │                             pregame_game_card.schema.json`)
                          │                             — currently
                          │                             unpopulated
                          │                             placeholders with
                          │                             no producing
                          │                             engine
                          │
                          └─> `projected_run_differential` — a
                              **future, not-yet-existing** derived
                              field (analogous to the schema's existing
                              `projected_home_runs` /
                              `projected_away_runs` /
                              `projected_total`) that a future run-line
                              model would need to produce before
                              `home_minus_1_5_probability` /
                              `away_minus_1_5_probability` can be
                              computed from it. It is named here as a
                              target lineage node, not claimed as an
                              existing column.
```

**Explicit distinction — do not conflate these two concepts:**
"A game was decided by 2+ runs" (the factual outcome captured by
`won_by_2_plus` / `lost_by_2_plus` / `target_team_win_by_2_plus`, all
computed purely from `run_differential`) is **not the same statement**
as "the sportsbook favorite covered a -1.5 run line." The repository's
own `covered_minus_1_5_result` field on `team_outcome_classifier.py`
happens to share the same `run_differential >= 2` threshold as
`won_by_2_plus` for the favorite side, but a run *line* is a market
construct (favorite must win by 2+, underdog covers by losing by 1 or
winning outright) tied to a specific book/price, whereas
`target_team_win_by_2_plus` is a market-independent, symmetric factual
target computed identically for both teams from the final score alone.
A future run-line prediction model must not assume these are
interchangeable just because they overlap arithmetically for the
favorite.

## 1b. Totals / scoring-shape learning-objective lineage (first-class, independent target family)

Totals (and the scoring-shape distribution underneath a total) are a
**first-class Baseball Brain target family**, traced end to end here in
the same way as Section 1a traces moneyline/run-margin. Totals must
remain **structurally independent** of moneyline and run-margin — no
totals column is derived from, or merged into, `target_team_win*`,
`run_margin`, `home_margin`/`away_margin`, or `margin_*_plus` — while
still sharing the same upstream pregame identities and matchup facts
(the same `pregame_team_identities` / `pregame_identity_matchups` /
`clean_bullpen_pregame_facts` / `batter_pregame_snapshots` nodes from
Section 2 feed both families; only the postgame target layer diverges).

```
pregame baseball state (Section 2 chain: pregame_team_identities,
pregame_identity_matchups, clean_bullpen_pregame_facts,
batter_pregame_snapshots)
  │
  ├─> [FUTURE model output] projected_home_runs
  ├─> [FUTURE model output] projected_away_runs
  │     (both already named as placeholder fields on
  │      `predictions.projected_home_runs` / `predictions.
  │      projected_away_runs`, `schemas/pregame_game_card.schema.json`
  │      — currently unpopulated, no producing engine)
  │         │
  │         └─> [FUTURE model output] projected_total_runs
  │             (`predictions.projected_total` on the same schema —
  │             unpopulated placeholder)
  │                 │
  │                 ├─> [FUTURE model output] scoring-shape probabilities
  │                 │   — no schema field or engine exists yet for a
  │                 │   distribution over `total_run_bucket` categories
  │                 │   (see below); named here as a required future
  │                 │   node, not claimed as existing
  │                 │
  │                 └─> [FUTURE model output] over/under probabilities
  │                     (`predictions.over_probability` / `predictions.
  │                     under_probability` on the same schema —
  │                     unpopulated placeholders, no producing engine)
  │
  └─────────────────────────────────────────────────────────────────┐
                                                                      │
master_game_database.parquet (canonical final scores)                │
  └─> atlas/learning/factual_target_builder.py                       │
        build_game_targets (home_score, away_score, game_total_runs) │
        build_team_game_targets (team_runs, opponent_runs,           │
          target_team_scored_3_or_less, target_team_scored_exactly_4,│
          target_team_scored_5_plus — per-team-game grain, already   │
          frozen production columns)                                │
        │                                                            │
        └─> atlas/learning/totals_target_builder.py (NEW, this       │
            change) — build_total_runs_targets:                     ◄┘
              reads only the frozen columns above, never mutates
              them, and never touches target_team_win*/run_margin.
              Produces a brand-new, independent game-level table:
                - home_runs_scored, away_runs_scored, actual_total_runs
                - home_team_scored_3_or_less / _exactly_4 / _5_plus
                - away_team_scored_3_or_less / _exactly_4 / _5_plus
                - low_scoring_game, high_scoring_game,
                  extreme_high_scoring_game
                - total_run_bucket (low / average / high / extreme_high)
              │
              └─> actual_total_runs (factual, postgame)
                    │
                    └─> postgame Game Story explanation and learning
                        (see below — future canonical artifact)
```

**Total-run bucket boundaries are data-derived, not arbitrary.** They
come from the canonical 2024 completed-game `game_total_runs`
distribution (2,428 games; `atlas_reference/samples/games/
data__game_intelligence__factual_learning_targets__2024__factual_game_learning_targets.parquet.games.parquet`):
25th percentile = 5, median = 8, 75th percentile = 11, 90th percentile
= 15. `total_run_bucket` therefore separates `low` (≤5, "approximately
4-run games" territory), `average` (6-11), `high` (≥12, matching this
family's explicit "12+-run games" requirement — one run above the
75th-percentile cut point), and `extreme_high` (≥15, the 90th-percentile
cut point). This is exactly the boundary the dependency graph must
support analysis across: what separates ~4-run games from 12+-run
games.

**Do not reduce totals learning to a single over/under 8.5 label.**
`actual_total_runs` (continuous), `total_run_bucket` (categorical
scoring shape), and the per-side `*_team_scored_*` facts above are all
preserved as first-class outcomes; a market over/under line is never
required to compute any of them (`market_line_used` is always `False`
on `totals_target_builder.py`'s output).

**Structured Game Story WHY-classification for scoring (totals side):**
per Section 1's five-section table, `game_story_record` is **not
covered — future canonical artifact** for any target family. For
totals specifically, that future Game Story must classify *why*
scoring was low or high, covering at minimum:

- starter-driven suppression or damage
- bullpen-driven suppression or collapse
- lineup quality and missing hitters
- pitch-type and handedness matchup effects
- walks, strikeouts, and traffic creation
- contact quality and home-run damage
- park, weather, and umpire context
- defense and catcher effects
- late-inning scoring
- extra-inning inflation
- one-team blowout versus two-sided scoring
- sustained scoring versus one anomalous inning

Supporting engines partially exist for some of these categories
(`atlas/game_intelligence/game_flow_fact_table.py`,
`scoring_state_timeline.py`, `scoring_event_roles.py`,
`team_game_flow.py`, `response_recovery.py`, `lead_protection.py`), but
none assemble a structured, causal "why was this game's total
low/high" narrative today. This is flagged for review, not started
here, consistent with Section 1's treatment of `game_story_record` for
every other target family.

**Six new readiness rows, added to `atlas/audit/coverage_matrix.py`'s
`COVERAGE_ROWS`/`MODULE_ONLY_ROWS` by this change**, give totals
learning readiness coverage separate from generic
`model_artifacts`/`frozen_predictions`:

| Row | What it evidences | Current status |
|---|---|---|
| `total_runs_targets` | Factual total-runs targets (`actual_total_runs`, `total_run_bucket`, etc.) | **Existing** — `atlas/learning/totals_target_builder.py` |
| `scoring_shape_classification` | `total_run_bucket` / per-side scoring-shape facts | **Existing** — same module |
| `projected_team_runs` | `projected_home_runs`/`projected_away_runs` model | **Not ready — no module exists**; schema placeholders only |
| `projected_game_total` | `projected_total_runs` model | **Not ready — no module exists**; schema placeholder only |
| `over_under_model_readiness` | `over_probability`/`under_probability` model | **Not ready — no module exists**; schema placeholders only |
| `team_total_model_readiness` | Per-team total (team-run) prediction model | **Not ready — no module exists** |

All six map to the pre-existing `"totals"` focus-area keyword in
`atlas/audit/repository_inventory.py`'s `FOCUS_AREA_KEYWORDS` (already
present before this change, previously unused by any coverage row),
so the readiness audit now discovers
`atlas/learning/totals_target_builder.py` automatically.

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

### 2a. Authoritative published-schedule layer (first-class dependency)

`atlas/schedule/mlb_schedule_reference.py` is anchored here as a
first-class, independent software dependency — not a derived/internal
table like `master_game_database.parquet`. It builds a canonical,
one-row-per-`gamePk` reference dataset **only** from the published MLB
Stats API `/schedule` endpoint, and is (per its own module docstring and
`atlas/audit/schedule_source_assessment.py`) the only source of
pregame-safe schedule/series-context facts recognized by the ATLAS
historical readiness audit. Its published `CANONICAL_FIELDS` cover:

- official `game_pk` (the only durable game identifier used; team/player
  *names* are carried as labels only and never used as a durable key)
- `game_type_code` / `season_segment` (regular season vs. postseason vs.
  all-star/spring/exhibition, via `GAME_TYPE_SEASON_SEGMENT`)
- game date (`game_date_utc`, `official_date`)
- scheduled start time (carried in the raw payload's game-datetime
  field feeding `game_date_utc`)
- home and away teams (`home_team_id`/`home_team_name`,
  `away_team_id`/`away_team_name`)
- venue (`venue_id`, `venue_name`)
- published series length and series game number (`games_in_series`,
  `series_game_number`, `series_description`)
- doubleheaders (`double_header_code`, `game_number`)
- postponements, suspensions (`status_code`, `coded_game_state`,
  `detailed_state`, `game_state_category`, via
  `DETAILED_STATE_CATEGORY`, including `postponed` / `suspended` /
  `cancelled` states)
- makeup games — a postponed game that is later replayed keeps the same
  `gamePk` under MLB's own scheduling model, so de-duplicating by
  `gamePk` (via `_STATE_CATEGORY_PRIORITY`) is sufficient to avoid
  double-counting a makeup game as a second real-world game
- rest and derived travel/time-zone context — **not** published fields
  themselves; these are explicitly *derived* downstream from this
  layer's `game_pk` + `game_date_utc` + venue sequence per team (see the
  `rest` / `travel` rows in Section 3), not invented as new schedule
  columns on this module

**Preserved rule (unchanged, load-bearing):** published series length
(`games_in_series` / `series_game_number`) comes only from the
published MLB Stats API schedule response and is **never** inferred
from completed-game counts, `status` completion, or
`master_game_database.parquet` history. This module enforces that rule
by construction (it never reads score/result data), and
`atlas/audit/schedule_source_assessment.py` enforces it for the rest of
the audit pipeline. Any future rest/travel builder must derive its
inputs from this schedule layer's `game_pk`/date/venue sequence, not
from result data, to preserve the same rule.

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
| Series context | `series_game`, `published_series_context` | `atlas/schedule/mlb_schedule_reference.py` (first-class published-schedule builder — see Section 2a) plus `atlas/audit/schedule_source_assessment.py`, which implements the no-leakage rule over it | Governed by the same leakage rule as `published_schedule`; data presence unverified | **Existing rule and existing builder module, live-fetch data presence unverified.** |
| Run margin / run-line factual target | `team_margin` (existing tag) | `atlas/game_intelligence/outcome_classifier.py`, `team_outcome_classifier.py`, `atlas/learning/factual_target_builder.py`, `atlas/validation/target_resolution.py` — see Section 1a for the full lineage | `target_team_win_by_2_plus` present in 2024 univariate-evidence-discovery and candidate-integrity schemas; consumed by `concept_validation_2025.py` | **Existing target + validation lineage.** No dedicated run-line (spread) *prediction model* exists — see Section 1a and Section 5 item 5. |
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
5. **Run-margin/run-line status is more advanced than "not found"**: the
   factual `target_team_win_by_2_plus` target, its 2024 concept
   lineage, and its 2025 blind-validation resolution path all already
   exist (Section 1a). What is genuinely missing is a *dedicated
   run-line prediction-model module* that consumes that lineage and
   populates `home_minus_1_5_probability` / `away_minus_1_5_probability`
   on the Pregame Game Card — that is a future-model gap, not a
   missing-target gap.

## 5a. Explicit readiness coverage added by this document

The following readiness dimensions are named here explicitly, each
mapped to the section that defines its current status, so none of them
can be silently dropped from future readiness audits:

| Readiness dimension | Current status | Where defined |
|---|---|---|
| Historical schedule completeness | Existing builder (`mlb_schedule_reference.py`) and existing no-leakage rule (`schedule_source_assessment.py`); live per-season completeness against the published API is unverified in this offline session | Section 2a |
| Run-margin factual-target readiness | **Existing** — `target_team_win_by_2_plus` materializes cleanly from canonical `won`/`run_differential` via `atlas/validation/target_resolution.py`; upstream `master_game_database` provenance still `unknown` per `ATLAS_DATA_ALIGNMENT_AUDIT.md` Section 2 | Section 1a |
| Run-line prediction-model readiness | **Not ready — no module exists.** Only the factual target and its validation lineage exist; no model trains or serves `home_minus_1_5_probability`/`away_minus_1_5_probability` today | Section 1a, Section 5 item 5 |
| `game_anatomy` | **Not covered — future canonical artifact**, per Section 1's five-section table | Section 1 |
| `game_story_record` (structured, WHY) | **Not covered — future canonical artifact** | Section 1 |
| `learning_observations` (per game) | **Partially covered** — season/concept-grain evidence exists; no per-game record | Section 3 |
| `identity_update_log` (per game) | **Partially covered** — pregame-direction identity timeline exists; no postgame per-game delta record | Section 3 |
| Factual total-runs targets | **Existing** — `atlas/learning/totals_target_builder.py`, structurally independent of moneyline/run-margin | Section 1b |
| Scoring-shape classification | **Existing** — `total_run_bucket` and per-side scoring facts, same module | Section 1b |
| Projected team runs (`projected_home_runs`/`projected_away_runs`) | **Not ready — no module exists**; schema placeholders only | Section 1b |
| Projected game total (`projected_total_runs`) | **Not ready — no module exists**; schema placeholder only | Section 1b |
| Over/under prediction-model readiness | **Not ready — no module exists**; schema placeholders only | Section 1b |
| Team-total prediction readiness | **Not ready — no module exists** | Section 1b |

## 6. What this graph does not do

- It does not authorize uploading anything. See
  `docs/ATLAS_2024_BRAIN_UPLOAD_MANIFEST.md` for scoped upload
  recommendations derived from this graph.
- This session *does* add six totals-specific rows to
  `atlas/audit/coverage_matrix.py`'s `COVERAGE_ROWS`/`MODULE_ONLY_ROWS`
  (Section 1b) and the accompanying `totals_target_builder.py` module,
  because those were concrete, scoped, non-frozen additions requested
  directly. It does **not** add `park_factors`, `handedness_splits`,
  `manager_tendencies`, `roster_transactions`, `game_story`,
  `game_anatomy`, `learning_observations`, or `identity_updates` as new
  tracked rows — those remain a recommended follow-up, not performed in
  this session.
