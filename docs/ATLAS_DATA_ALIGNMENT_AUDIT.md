# ATLAS Data Alignment Audit

Status: **initial repository-evidence audit**, produced without any Cloud
Storage or Google Drive access from this environment. It is the first
deliverable of the "complete repository alignment and historical
reconstruction audit" requested before any 2024 rebuild work begins.

This document is **read-only in intent**: it makes no changes to
`atlas/`, `data/`, or any Cloud Storage object, and it does not run a
rebuild, a backtest, model training, or prediction generation. It only
records what was found and what remains unknown.

## 1. What could and could not be verified in this session

| Check | Result |
|---|---|
| `python -m pytest tests/test_atlas_audit_*.py -q` | **124 passed**, 0 failed |
| `atlas.audit.repository_inventory.build_repository_inventory(".")` (offline, no cloud) | Ran successfully; results below |
| Live `gcloud storage objects list gs://atlas-mlb-data-brian-4817/**` | **Not possible from this session** — `gcloud auth list` shows no credentialed account. This is expected: the repository's own design puts live-bucket access behind the `atlas-historical-readiness-audit.yml` GitHub Actions workflow (Workload Identity Federation), which must be dispatched from GitHub Actions, not from this sandbox. **Recommendation: dispatch `atlas-historical-readiness-audit.yml` via `workflow_dispatch` and attach its `artifacts/audits/` output to this audit** — that is the only channel through which the real bucket contents can be verified, and it already exists; it does not need to be built. |
| Local `data/` directory | Does not exist in this clone. `.gitignore` excludes `data/` and `reports/` by design — this repository intentionally ships code, contracts, and a compact reference pack only (`atlas_reference/compact_manifest.json`: `"production_data_included": false"`). |

Because the live bucket could not be queried here, **no dataset in this
document is classified as `canonical` on the strength of this session
alone.** Classifications below are either (a) inherited from evidence
already recorded in-repo (frozen phase-completion docs, the compact
manifest, the audit tooling's own design assumptions) or (b) marked
`unknown — requires live audit run`.

## 2. A discrepancy that must be resolved before any 2024 rebuild work

Two different canonical-storage locations are referenced by different
parts of the repository, and nothing in the repo reconciles them:

1. `config/atlas_config.py` and most of `atlas/` (13 modules flagged by
   `repository_inventory` as `colab_or_drive_dependency=true`, e.g.
   `atlas/game_intelligence/game_flow_fact_table.py`,
   `atlas/game_intelligence/scoring_state_timeline.py`,
   `atlas/game_intelligence/lead_protection.py`,
   `atlas/game_intelligence/response_recovery.py`,
   `atlas/config/paths.py`) assume production data lives at
   `/content/drive/MyDrive/Project_Atlas/...` — a Google Drive path only
   reachable from a Colab runtime.
2. `.github/workflows/atlas-historical-readiness-audit.yml`,
   `.github/workflows/atlas-cloud-data-test.yml`, and
   `.github/workflows/google-cloud-auth-test.yml` assume production data
   (or at least the four master tables) lives in
   `gs://atlas-mlb-data-brian-4817` via Workload Identity Federation.
3. `docs/AUTOPILOT_EXECUTION_LEDGER.md` (the most recent status ledger)
   states plainly: *"All Phase 2E.5A+ work depends on production
   artifacts that exist only at `/content/drive/MyDrive/Project_Atlas/...`
   in the Colab runtime. They are not present in this sandboxed git
   clone."* It does not mention the GCS bucket at all.

**This is an unresolved lineage question, not an assumption to paper
over.** Before the dependency graph in
`docs/ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md` can be trusted end to
end, someone with access to both locations must confirm:
- Is `gs://atlas-mlb-data-brian-4817` a mirror/export of the Drive data,
  a separate/newer canonical copy, or a partial subset (e.g. only the
  four "known master files")?
- Which location is authoritative for the 2024 learning season today?

**A partial, already-documented resolution path exists and should be
used first.** `docs/DEV_DATA_BUNDLE.md` and
`atlas_reference/dev_data_bundle_required_artifacts.json` describe a
versioned, checksum-verified GitHub Release "data bridge" that packages
a named allowlist of real production files straight out of
`/content/drive/MyDrive/Project_Atlas` (Colab/Drive) for use outside
Drive via `ATLAS_DATA_ROOT`. That registry currently lists **13**
required artifacts (e.g. `clean_bullpen_pregame_facts_2025`,
`batter_pregame_snapshots_2025`, `pitcher_pregame_snapshots_2025`,
`historical_starting_lineups`, `game_anomaly_registry`), each with a
`primary_key` and `consuming_builder` already recorded — this is a more
authoritative, code-verified dependency source than anything this audit
can infer from `atlas_reference/manifests/table_catalog.json` alone, and
should be treated as the current best evidence of Drive-side lineage.
It does **not** mention `gs://atlas-mlb-data-brian-4817` at all, which
reinforces that the GCS-bucket audit workflow is either a separate,
newer effort or an incomplete migration — not something this registry
already accounts for.

Until answered, this audit treats **both** locations as
`provenance_status: unknown` for anything beyond the four files
`cloud_inventory.py` already knows how to name
(`master_game_database.parquet`, `master_pitch_database.parquet`,
`master_game_database_metadata.json`, `team_game_state.parquet`).

## 3. Repository-code inventory (verified, offline, this session)

Produced by `atlas.audit.repository_inventory.build_repository_inventory`
run directly against this checkout on 2026-07-20 (no fabricated entries
— every path below exists in the repo as of that run). These counts are
a point-in-time snapshot, not a frozen contract: re-run the same
function (offline, no cloud access required) to refresh them as the
codebase evolves; do not treat this document's numbers as current
without re-verifying.

- **150** Python modules under `atlas/`, **7** under `scripts/`, **43**
  test modules, **4** workflow files, **0** notebooks in this clone.
- **19 duplicate-symbol groups** — the same function/class name defined
  in more than one module, a signal of parallel/duplicated logic that
  should be resolved before it is relied on for a rebuild, e.g.:
  - `load_master_games` defined in `atlas/history/history_engine.py`,
    `atlas/pitchers/pitcher_engine.py`, **and** a backup file
    `atlas/pitchers/backups/pitcher_engine_v1_20260711T024923Z.py`.
  - `build_game_targets` / `build_team_game_targets` defined in both
    `atlas/learning/backtest_target_builder.py` and
    `atlas/learning/factual_target_builder.py`.
  - `condition_mask` defined in five separate `atlas/learning/*.py`
    modules.
  - Full list in `artifacts/audits/repository_inventory.json` when the
    tool is re-run (not committed here, per the "Reports" `.gitignore`
    convention already used for `atlas-historical-readiness-audit.yml`
    output).
- **Zero** modules under `atlas/` matched the literal focus-area keyword
  `run_line` (or `moneyline`, `totals`, `player_props`) in a filename
  scan. **Correction to an earlier draft of this audit:** this only
  means no module is *named* `run_line`-something — it does **not**
  mean run-margin/run-line learning is absent from the repository.
  `atlas/game_intelligence/team_outcome_classifier.py`,
  `atlas/learning/factual_target_builder.py` /
  `backtest_target_builder.py`, and `atlas/validation/
  target_resolution.py` already implement the full factual 2+ run-margin
  target (`target_team_win_by_2_plus`, resolved from canonical `won` /
  `run_differential`) plus its 2024 concept lineage and its 2025
  blind-validation resolution path — see
  `docs/ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md` Section 1a for the
  complete, code-verified lineage. What is genuinely absent is a
  **dedicated run-line *prediction-model* module** — nothing under
  `atlas/` trains or serves `home_minus_1_5_probability` /
  `away_minus_1_5_probability` on the Pregame Game Card today. Those are
  two different claims and must not be conflated.
- Similarly, **zero** modules matched
  `master_game_builder`, `master_pitch_builder`, `team_game_state`,
  `offense`, `contact`, `discipline`, `team_pitching`, or
  `feature_lineage`. Either these builders live exclusively in the
  Colab/Drive codebase not mirrored into this repo, or they genuinely do
  not exist yet as permanent modules. This must be confirmed, not
  assumed either way.
- **13** modules have a direct Colab/Drive dependency (see list above),
  confirming that a real 2024 rebuild cannot run inside this repo/CI as
  currently structured without either (a) porting those modules to read
  from the GCS bucket, or (b) running the rebuild inside a Colab runtime
  with Drive mounted — a design decision, not a code bug.

## 4. Existing audit tooling already implements the requested trust model

`atlas/audit/coverage_matrix.py` already tracks 27 named rows
(`COVERAGE_ROWS`) across the three seasons, each independently scored on
five dimensions (`data_presence`, `source_completeness`,
`provenance_status`, `temporal_availability`, `pregame_safety`):
`published_schedule`, `game_identifiers`, `scheduled_first_pitch`,
`final_scores`, `pitch_by_pitch`, `plate_appearances`,
`batted_ball_data`, `starters`, `bullpen_usage`, `lineups`, `injuries`,
`weather`, `venue`, `umpire`, `rest`, `travel`,
`published_series_context`, `opening_market`, `closing_market`,
`team_memories`, `player_memories`, `identities`, `concept_discovery`,
`concept_validation`, `model_artifacts`, `frozen_predictions`,
`frozen_pregame_cards`.

This is the correct existing framework to extend, not replace. Rows
**not yet tracked** at all (`park_factors` as distinct from `venue`,
`manager_tendencies`, `handedness_splits`, `pitch_type_interactions`,
`roster_transactions`, `game_story`, `game_anatomy`,
`learning_observations`, `identity_updates`) are the subject of
`docs/ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md`, Section 3.

Running this session's actual result of `_generic_dataset_row` /
`_module_row` for every `COVERAGE_ROWS` entry requires the live cloud
inventory (Section 1) and the local `dataset_profile` output for
`master_game_database.parquet` / `master_pitch_database.parquet`,
neither of which was retrievable in this offline session. **The
coverage matrix must be re-run inside the `atlas-historical-readiness-audit`
workflow to populate real values** — this document does not fabricate
them.

## 5. Dataset trust classification (what can be said without live access)

| Dataset / artifact | Evidence available here | Classification |
|---|---|---|
| `atlas_reference/manifest.json`, `manifests/table_catalog.json`, `game_data_catalog.json`, `player_data_catalog.json`, `relationship_map.json`, `schemas/*.schema.json`, `samples/*` | In-repo, versioned, explicitly `full_source_hashing: false` (hashing was disabled when generated for cost reasons) | **Reference pack, not canonical production data.** Useful as a schema/shape reference only; every row-count, key, and lineage claim it implies must be re-verified against live sources before being trusted for a rebuild. |
| Four "known master files" (`master_game_database.parquet`, `master_pitch_database.parquet`, `master_game_database_metadata.json`, `team_game_state.parquet`) | `atlas/audit/cloud_inventory.py` already names these as the expected canonical set in `gs://atlas-mlb-data-brian-4817/data/master/` | **Unknown — requires live audit run.** The tooling to check them (presence, hash, row count) already exists and is unit-tested; it has simply not been executed against the real bucket in this session. |
| Frozen 2024 concept definitions (2,138 concepts / 4,276 members per `docs/ATLAS_BRAIN_PHASE_2E_4G_IMMUTABLE_CONCEPT_FREEZE.md`) | Documented as frozen and hash-fingerprinted in a prior session | **Trusted as a frozen contract per its own governance rules** (not to be re-verified by re-deriving it — that would violate its own freeze rule) but **its upstream inputs** (2024 identity/bullpen/lineup artifacts) are themselves `unknown — requires live audit run`. |
| Phase 2E.5A 2025 blind-validation evidence matrix | `docs/ATLAS_BRAIN_PHASE_2E_5A_2025_VALIDATION_INPUT_READINESS.md` records `Global readiness gate passed: no` | **Incomplete by the project's own record**, not stale or corrupted — simply not yet built. |
| Google Drive production artifacts (`/content/drive/MyDrive/Project_Atlas/...`) | Referenced by 13 modules and the ledger; not reachable from this session or from GitHub Actions | **Unknown / unreachable from CI.** If this is still the authoritative 2024 store, no GitHub Actions-based audit (including the new WIF-based one) can currently see it. This must be resolved per Section 2 before claiming any GCS-based audit result "proves" 2024 readiness. |

## 5a. Explicit readiness coverage (this document's contribution)

The dependency-graph document (`ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md`
Section 1a, 2a, 5a) traces the full lineage and rules for these
dimensions; this audit records their present-day readiness status so it
is not lost between documents:

| Readiness dimension | Status recorded here |
|---|---|
| Historical schedule completeness | `published_schedule` is already a tracked `COVERAGE_ROWS` row, and `atlas/schedule/mlb_schedule_reference.py` is the code-verified authoritative builder for it (Section 4 above, dependency-graph Section 2a). Live per-season completeness against the published MLB Stats API is `unknown — requires live audit run`, same as every other row in Section 5. |
| Run-margin factual-target readiness | **Existing**, per the corrected Section 3 finding above — `target_team_win_by_2_plus` resolves cleanly from canonical `won`/`run_differential`; blocked only by the same upstream `master_game_database` provenance question as everything else in Section 2. |
| Run-line prediction-model readiness | **Not ready.** No module trains or serves a standalone run-line prediction; only the factual target and its validation lineage exist (dependency-graph Section 1a). |
| `game_anatomy` | **Not covered — future canonical artifact.** See dependency-graph Section 1. |
| `game_story_record` | **Not covered — future canonical artifact.** See dependency-graph Section 1. |
| `learning_observations` (per game) | **Partially covered** — season/concept-grain evidence exists (`atlas/learning/evidence_consolidation_engine.py`); no per-game record exists. |
| `identity_update_log` (per game) | **Partially covered** — pregame-direction identity timeline exists (`pregame_team_identity_timeline.py`); no postgame per-game delta record exists. |

## 6. Immediate next actions (no rebuild, staging-only)

1. Resolve the Drive-vs-GCS-bucket lineage question (Section 2) with
   whoever manages both locations.
2. Dispatch `atlas-historical-readiness-audit.yml` to get a live
   `cloud_object_inventory.json`, `historical_coverage_matrix.json`, and
   `historical_readiness_report.json` from the real bucket, and attach
   those artifacts to this audit as the authoritative Section 5 update.
3. Resolve the 19 duplicate-symbol groups in `atlas/` (Section 3) before
   any of them are relied on as the single source of truth for a rebuild
   input.
4. Only after 1–3: proceed to
   `docs/ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md` and
   `docs/ATLAS_2024_BRAIN_UPLOAD_MANIFEST.md` for what is actually
   missing versus what merely needs to be re-verified.
