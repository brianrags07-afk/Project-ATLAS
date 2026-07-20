# ATLAS 2024/2025 Upload Manifest (Recommendations)

This manifest lists **specific, named datasets** recommended for upload,
each justified by a concrete gap identified in
`docs/ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md`. It intentionally does
**not** ask for entire folders "just in case." Every row below states
why the dataset is needed, its current status, its target season(s),
and its destination — always a **staging** path, never a canonical
cloud path.

No upload should occur until:
1. `docs/ATLAS_DATA_ALIGNMENT_AUDIT.md` Section 2 (Drive-vs-GCS-bucket
   lineage) is resolved, and
2. `atlas-historical-readiness-audit.yml` has been run at least once
   against the real bucket to confirm what is already present (some
   rows below may turn out to already exist and simply need
   provenance evidence attached, not a fresh upload).

All destination paths below are proposals under a new
`staging/uploads/` prefix, distinct from `data/master/`, `data/daily/`,
etc. so nothing here is ever confused with or capable of overwriting an
existing canonical object.

## 1. Datasets to verify before uploading anything (may already exist)

| Dataset | Why needed | Current status | Season(s) | Action (not upload) |
|---|---|---|---|---|
| `master_game_database.parquet` | Root dependency for nearly every node in the dependency graph | `provenance_status: unknown` — not confirmed present/canonical in this session | 2024, 2025 | Verify via live audit run first; do not re-upload speculatively. |
| `master_pitch_database.parquet` | Root dependency for pitch-type interaction, handedness, and game-flow nodes | `provenance_status: unknown` | 2024, 2025 | Same as above. |
| `team_game_state.parquet` | Feeds team-level identity/evidence engines | `provenance_status: unknown` | 2024, 2025 | Same as above. |
| Frozen 2024 concept definitions (2,138 concepts) | Already governed as a frozen contract | Documented as frozen/hash-fingerprinted | 2024 | Never regenerate or mutate the frozen definition/member registries. **If** the exact checksum-verified frozen artifacts are confirmed absent from `gs://atlas-mlb-data-brian-4817` and a downstream consumer (e.g. 2025 blind validation) genuinely needs them, the verified files may be **copied as-is** (byte-identical, checksum-matched against the frozen fingerprint already recorded in `docs/ATLAS_BRAIN_PHASE_2E_4G_IMMUTABLE_CONCEPT_FREEZE.md`) to a staging path (e.g. `staging/uploads/frozen_2024_concepts/`) with full lineage metadata (source path, checksum, copy timestamp, requesting consumer). This is a **verified copy**, not a re-derivation, and is not itself a promotion to canonical — later promotion out of staging requires separate, explicit approval. Do not upload if the canonical GCS copy is already confirmed present. |

## 2. New canonical datasets recommended for upload (genuinely absent, per the dependency graph)

Each of these was searched for by name and by keyword across
`atlas_reference/manifests/table_catalog.json` (580 cataloged tables)
and returned **zero matches**, and has no corresponding builder under
`atlas/`. These are not "files that were forgotten" — they are datasets
that, per repository evidence, have never existed in this project.

| Dataset (proposed name) | Why needed (graph node it unblocks) | Expected grain | Season(s) needed | Staging destination |
|---|---|---|---|---|
| `park_factors` [^park-factors] | `environment.park_factors` on the Pregame Game Card; currently only `venue` (name/location) exists, not park run/HR/handedness effects | Per venue, per season | 2024 (learning), 2025 (blind validation) | `staging/uploads/park_factors/{season}/` |
| `weather` | `environment.weather` on the Pregame Game Card; tracked as a `COVERAGE_ROWS` row in `atlas/audit/coverage_matrix.py` but zero data found | Per game (pregame forecast) and ideally per-game actual conditions, each with its own `source_timestamp_utc` per the Game Card contract's source-timestamp rule | 2024, 2025 | `staging/uploads/weather/{season}/` |
| `umpire_tendencies` (assignments + historical tendency profile) | `environment.umpire` on the Pregame Game Card; tracked as a `COVERAGE_ROWS` row, zero data found | Two sub-datasets needed: (a) per-game umpire assignment with assignment timestamp (for pregame-safety proof — umpire is often not confirmed far in advance), (b) per-umpire historical strike-zone/tendency profile computed only from prior completed games | 2024, 2025 | `staging/uploads/umpire_assignments/{season}/`, `staging/uploads/umpire_tendency_profiles/{season}/` |
| `travel_context` | `schedule_context.*_travel_context` on the Pregame Game Card; no venue geo/timezone reference exists anywhere in `atlas_reference/` | Static reference table (venue → lat/long/timezone), plus a per-game derived distance/timezone-change field | One-time reference table (not season-specific) + per-game derived field for 2024, 2025 | `staging/uploads/venue_reference/`, `staging/uploads/travel_context/{season}/` |
| `handedness_splits` | Matchup features on `starters.*` and `lineups.*.batting_order[].handedness`; likely derivable from `master_pitch_database.parquet` rather than needing a fresh upload | Per player, per season, split by opposing-hand | 2024, 2025 | Prefer building from existing pitch data once confirmed present; only upload if the raw at-bat data is confirmed absent. |
| `manager_tendencies` | No current dataset or engine; named explicitly in the task as a required learning objective | Per manager, per season: bullpen-usage patterns, pinch-hit/steal aggressiveness by game state, lineup-construction habits | 2024, 2025 | `staging/uploads/manager_tendencies/{season}/` |
| `roster_transactions` | Gates pregame-safety of `starters`/`lineups`/`bullpen` sections — "who was even on the active roster on this date" is a pregame fact today, not a postgame one | Per transaction event (call-up, IL move, trade, DFA, release) with an authoritative transaction timestamp | 2024, 2025 | `staging/uploads/roster_transactions/{season}/` |

[^park-factors]: Park factors are typically computed on a rolling
multi-year basis, so 2021-2025 history may be needed to compute a
stable 2024/2025 factor. Confirm methodology (raw components vs.
pre-computed factor) before uploading.

## 3. Datasets that need a builder, not a fresh upload

| Dataset | Why | Source it should be built from | Action |
|---|---|---|---|
| `rest` (days of rest per team/pitcher per game) | Tracked as a `COVERAGE_ROWS`/`DYNAMIC_PREGAME_ROWS` row; derivable from schedule gaps | `atlas/schedule/mlb_schedule_reference.py` (published `game_pk`/`game_date_utc`/venue sequence per team) — not from completed-game results, to preserve the published-series-length rule in `ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md` Section 2a | Build, do not upload — once schedule-layer live-fetch coverage is confirmed (Section 5a below). |
| Travel / time-zone context (derived) | Same schedule-derived rule as `rest` | `atlas/schedule/mlb_schedule_reference.py` venue sequence + the new `travel_context` venue-geo reference (Section 2) | Build once both inputs exist — not an upload task for the derived field itself. |
| Pitch-type interaction features | Raw pitch-level data likely exists | `master_pitch_database.parquet` via `atlas/pitchers/v2/pitch_table.py` | Build, do not upload. |
| `game_anatomy` / `game_story_record` / `learning_observations` / `identity_update_log` | New contract/schema design work, not a data-upload problem (see `ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md` Section 1 and 3) | Assembled from existing engines (`game_flow_fact_table.py`, `outcome_classifier.py`, `evidence_consolidation_engine.py`, `pregame_team_identity_timeline.py`) plus the new datasets above | Schema/contract design first, then a builder — not an upload task at all. |
| Run-line prediction model | Named future model in `ATLAS_BASEBALL_BRAIN_DEPENDENCY_GRAPH.md` Section 1a; the factual `target_team_win_by_2_plus` lineage it would train against already exists — this row is model/engine design work, not a data gap | Consumes `atlas/validation/target_resolution.py` output (2024 learning + 2025 blind validation) | Model design/implementation task, not an upload — explicitly out of scope for this documentation-only manifest. |

## 3a. Explicit readiness coverage (this manifest's contribution)

| Readiness dimension | Upload-relevant status |
|---|---|
| Historical schedule completeness | No upload needed — `atlas/schedule/mlb_schedule_reference.py` already fetches this live from the published API; only live-fetch verification (not a staged upload) is outstanding. |
| Run-margin factual-target readiness | No upload needed — `target_team_win_by_2_plus` is derived in-memory from already-present `won`/`run_differential` columns. |
| Run-line prediction-model readiness | Not an upload item — see the new row above; this is a future model-engineering task. |
| `game_anatomy`, `game_story_record`, `learning_observations`, `identity_update_log` | Not upload items — schema/contract design first, per the row above. |

## 4. Explicit non-requests

Per the task's instruction not to ask for folders blindly, this manifest
deliberately does **not** request:
- Any bulk `data/` folder upload.
- Any re-upload of `atlas_reference/` contents (already in-repo).
- Any raw Statcast/odds/injuries re-upload — `config/atlas_config.py`
  already names `RAW_DIR/{statcast,weather,odds,lineups,umpires,injuries}`
  as expected raw subdirectories; if any of those already contain
  weather/umpire/injury data under Drive, the correct next step is to
  **locate and inventory them** (Section 2 items may already be partly
  satisfied there) before uploading anything new. This should be the
  very first thing checked once Drive access is available, since it
  could eliminate several "new canonical dataset" rows above.

**Prefer the existing dev-data-bundle bridge over ad hoc uploads.**
`docs/DEV_DATA_BUNDLE.md` already defines a versioned, checksummed
GitHub Release packaging/bootstrap flow driven by an explicit allowlist
(`atlas_reference/dev_data_bundle_required_artifacts.json`). Any of the
Section 1 or Section 2 datasets that genuinely exist under
`/content/drive/MyDrive/Project_Atlas` should be added to that allowlist
and shipped through that existing mechanism rather than through a new,
one-off upload path — this keeps a single checksum-verified channel
instead of creating a second one.

## 5. Upload safety rules (apply to every row above)

- Every upload target is under `staging/uploads/...`, never
  `data/master/`, `data/daily/`, or any path already treated as
  canonical by `atlas/audit/cloud_inventory.py`'s
  `known_master_files_expected` set.
- No upload should overwrite an existing object at any path.
- Every uploaded dataset must arrive with (a) a source/retrieval
  timestamp per field where the Game Card contract requires one
  (weather, umpire, roster transactions all qualify), and (b) enough
  metadata to compute a hash for the eventual pipeline manifest
  (`schemas/pipeline_manifest.schema.json` `source_objects[].hash`).
  Uploads without this are `provenance_status: missing` on arrival, per
  the audit's own five-dimension model, and should not be treated as
  "canonical" just because they exist.
