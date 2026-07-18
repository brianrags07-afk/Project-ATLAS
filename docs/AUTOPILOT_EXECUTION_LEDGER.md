# ATLAS Autopilot Execution Ledger

This ledger is the persistent, authoritative record of autonomous
phase-by-phase execution state for Project ATLAS. It is updated after every
completed task. It supersedes narrative summaries in chat/task descriptions,
which may be stale or incomplete.

## Verified current checkpoint

Verified directly from repository evidence (not from filenames or comments
alone) on 2026-07-18:

- Contract pack commit `ca0462c` ("Add compact ATLAS development data and
  contract pack") is the latest commit on `main`-derived history before this
  ledger was created.
- `atlas_reference/compact_manifest.json` confirms `production_data_included:
  false` — this repository intentionally ships schemas, registries, and small
  representative samples only. Full production parquet artifacts live outside
  version control at `/content/drive/MyDrive/Project_Atlas/...` (Google Drive,
  Colab runtime) and are **not present** in this sandboxed clone.
- `docs/ATLAS_BRAIN_PHASE_2E_4G_IMMUTABLE_CONCEPT_FREEZE.md` states Phase
  2E.4G ("Immutable 2024 Concept Definition Freeze") **completed
  successfully**: 2,138 frozen concepts, 4,276 frozen members, registry
  SHA-256 fingerprints recorded and immutable.
- `docs/ATLAS_BRAIN_PHASE_2E_4F_TO_4G_CHECKPOINT.md` confirms the full
  2E.4F → 2E.4G chain (controlled concept formation, integrity adjudication,
  network independence, semantic governance, immutable freeze) is complete.
- `docs/ATLAS_BRAIN_PHASE_2E_5A_2025_VALIDATION_INPUT_READINESS.md` is the
  **newest, unfrozen** phase document. It records:
  - `Global readiness gate passed: no`
  - Next action: "Construct the governed 2025 canonical evidence matrix from
    the already-saved 2025 identity, bullpen and lineup-starter artifacts,
    then rerun this audit."
  - No corresponding permanent module or test exists yet under `atlas/` for
    2E.5A (unlike every completed 2E subphase, which has a frozen module in
    `atlas/game_intelligence/`, `atlas/learning/`, or `atlas/validation/` plus
    a matching `tests/test_*.py`).
- `docs/ATLAS_BRAIN_PHASE_2E_WORKING_CHECKPOINT.md` is **stale**: it only
  describes progress through Phase 2E.3A and is contradicted by the later,
  more specific phase documents (2E.3B through 2E.5A) which show
  substantially more completed work. This ledger treats the later, more
  specific documents as authoritative per repository evidence.
- `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/CHANGELOG.md`, and
  `docs/DECISIONS.md` were **stale** (Sprint-1-era, v0.1.0, dated
  2026-07-04), predating all Phase 2A–2E work. Refreshed in this session to
  reference the ATLAS Brain phase documents as the authoritative status
  source instead of duplicating fast-moving phase detail.

### Authoritative conclusion

- **Completed and frozen phases:** 1, 2A, 2B, 2C, 2D, and Phase 2E through
  2E.4G (immutable 2024 concept-definition freeze).
- **Current open phase:** Phase 2E.5A — 2025 blind-validation input
  readiness (canonical 2025 evidence matrix construction), not yet started
  at the permanent-module level.

## Completed phases

| Phase | Evidence | State |
|-------|----------|-------|
| 1 | `docs/ATLAS_BRAIN_PHASE_1_COMPLETE.md` | Frozen |
| 2A | `docs/ATLAS_BRAIN_PHASE_2A_COMPLETE.md` | Frozen |
| 2B | `docs/ATLAS_BRAIN_PHASE_2B_COMPLETE.md` | Frozen |
| 2C | `docs/ATLAS_BRAIN_PHASE_2C_COMPLETE.md` | Frozen |
| 2D | `docs/ATLAS_BRAIN_PHASE_2D_COMPLETE.md` | Frozen |
| 2E.1–2E.4G | `docs/ATLAS_BRAIN_PHASE_2E_4G_IMMUTABLE_CONCEPT_FREEZE.md`, `docs/ATLAS_BRAIN_PHASE_2E_4F_TO_4G_CHECKPOINT.md` | Frozen (2024 concept discovery, integrity, network independence, semantic governance, immutable freeze) |

## Current phase

**Phase 2E.5A — 2025 Blind Validation Input Readiness**

- Discovery source frozen (2024): 2,138 concepts / 4,276 members, immutable.
- Validation universe: 2025 season.
- Global readiness gate: **not yet passed**.
- No permanent `atlas/` module or `tests/test_*.py` yet implements 2E.5A logic.

## Remaining tasks (from repository evidence, in dependency order)

1. Construct the governed 2025 canonical evidence matrix builder as a
   permanent module (mirroring the pattern of `atlas/game_intelligence/
   pregame_identity_matchup_builder.py` and the Phase 2E.4A canonical core
   evidence matrix builder), consuming the already-saved 2025 identity,
   bullpen, and lineup-starter artifacts.
2. Add regression tests for the 2025 evidence matrix (grain, uniqueness,
   pregame safety, no-leakage, schema parity with the frozen 2024 evidence
   matrix contract).
3. Re-run the Phase 2E.5A readiness audit against the constructed 2025
   matrix; update `docs/ATLAS_BRAIN_PHASE_2E_5A_2025_VALIDATION_INPUT_READINESS.md`
   with a passing readiness gate once achieved.
4. Score the frozen 2024 concept definitions against 2025 evidence
   (blind validation) — governance requires this uses only frozen
   definitions; no new concepts, no threshold changes.
5. Produce Phase 2E.5B+ validation reports and, if governance approves,
   freeze Phase 2E as a whole with `ATLAS_BRAIN_PHASE_2E_COMPLETE.md`.
6. Only after Phase 2E is frozen: begin belief/probability/prediction
   fusion work described in `atlas/predictions/` and `atlas/learning/
   concept_belief_engine.py` using validated concepts.

## Dependencies

- All Phase 2E.5A+ work depends on production artifacts that exist only at
  `/content/drive/MyDrive/Project_Atlas/...` in the Colab runtime. They are
  **not present** in this sandboxed git clone and are excluded by design
  (`atlas_reference/compact_manifest.json`: `production_data_included:
  false`).
- `atlas_reference/` (manifest, schemas, registries, relationship map,
  samples) is the authoritative contract source for any new builder code and
  must be consulted before writing new data-builder logic, per
  `.github/copilot-instructions.md`.

## Risks

- Attempting to fabricate or synthesize 2025 production evidence to make
  tests pass would violate ATLAS non-negotiables (no fabricated data, no
  silent contract changes) and is explicitly prohibited.
- The stale top-level `docs/ROADMAP.md`/`BACKLOG.md`/`CHANGELOG.md` could
  mislead future contributors into thinking the project is at Sprint 1; they
  have been refreshed to point to the authoritative phase documents.

## Tests

Full suite run on 2026-07-18 (`python -m pytest tests/ -q`):

- **96 passed** — pure-logic/unit tests that do not require production Drive
  artifacts (concept governance rules, schema/contract checks operating on
  synthetic or bundled sample data).
- **111 failed** — every failure is a `FileNotFoundError` (or an `assert
  path.exists()` failure) for a path under
  `/content/drive/MyDrive/Project_Atlas/...`. No failure was caused by a
  logic defect in `atlas/` code; all are caused by the absence of production
  data in this sandboxed environment.
- **2 skipped** — explicitly self-skipping when 2024 identity artifacts are
  unavailable in this environment (by design, e.g.
  `tests/test_pregame_identity_matchup_builder.py`).
- No `TODO`, `FIXME`, or `NotImplementedError` markers exist anywhere under
  `atlas/`.

## Commits

- `ca0462c` — Add compact ATLAS development data and contract pack (prior
  checkpoint, confirmed frozen).
- (this session) — Add `docs/AUTOPILOT_EXECUTION_LEDGER.md`; refresh stale
  `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/CHANGELOG.md`.

## Blockers

**STOP CONDITION 3 — Required source data is unavailable.**

Constructing, testing, or freezing Phase 2E.5A (and all downstream
validation/belief/prediction work) requires the real 2025 pregame identity,
bullpen, and lineup-starter production artifacts. These exist only in the
Google Drive-backed Colab runtime this project normally executes in and are
intentionally excluded from this git repository. Without them:

- No new builder module can be verified against real data.
- No new regression test can assert real row counts, uniqueness, or audit
  outcomes without fabricating data, which is explicitly prohibited.

## Next action

When production data access is restored (Colab runtime with Drive mounted,
or an explicit governance decision to vendor a redacted sample of the 2025
evidence sources into `atlas_reference/samples/` the same way 2024 samples
were vendored): build the Phase 2E.5A canonical 2025 evidence matrix module
under `atlas/game_intelligence/` (or `atlas/validation/`, whichever the
relationship map indicates is authoritative), following the exact pattern of
the frozen Phase 2E.4A builder, then add regression tests, rerun the
readiness audit, and update this ledger.

## Recovery milestone (2026-07-18): repaired fabricated Phase 2E.1–2E.3A identity pipeline

**Failed commits identified**: `1a115c5`, `7491e69`, `ae6ba8d`, `0a22162`
(merged into `main` via `73b782d`, PR #1 "atlas-audit-and-report"). These
commits added:

- `atlas/game_intelligence/pregame_identity_source_registry.py` (Phase 2E.1)
- `atlas/game_intelligence/pregame_team_identity_timeline.py` (Phase 2E.2)
- `atlas/game_intelligence/pregame_identity_matchup_builder.py` (Phase 2E.3A)
- their three matching `tests/test_*.py` files.

**Root cause**: the prior implementation was built by guessing plausible
column names and a simplistic algorithm without access to the real ATLAS
production schema. It compiled and its own hand-written tests passed
(17 passed, 2 skipped) because those tests used synthetic fixtures matching
the same invented schema — a false-negative test smell, not evidence of
correctness. Once the compact `atlas_reference/` contract pack (commit
`ca0462c`) exposed the real schemas and fixtures, a column-by-column
comparison proved:

- Registry output columns were entirely fabricated (e.g.
  `identity_feature_name`, `min_lagged_days`) instead of the real
  `column, dtype, family, source_status, same_game_safe, requires_shift,
  historical_aggregation_allowed, non_null_rows, unique_values, reason`.
- Timeline computed a "last prior game" snapshot instead of a true
  strictly-prior-date expanding mean, and used the wrong output column
  prefix (missing `identity__expanding_mean__`).
- Matchup builder used `identity_edge_abs__` (reversed word order) instead
  of the real `identity_abs_edge__`, and was missing most of the real
  summary-diagnostic columns (`identity_sample_balance`,
  `identity_sample_confidence_score/label`, `both_teams_sample_*_plus`,
  edge sign/count summaries, `all_identity_edges_mirror`).

**Files rebuilt** (full rewrite, not patched — the design was based on
false assumptions and could not be salvaged incrementally):

- `atlas/game_intelligence/pregame_identity_source_registry.py`: rebuilt
  around a static, hardcoded 121-column classification table transcribed
  verbatim from the authoritative contract-pack fixture
  (`atlas_reference/samples/general/data__game_intelligence__
  pregame_identity_registry__2024__pregame_identity_source_registry.csv.
  sample.parquet`). `assert_matches_frozen_contract()` now rejects any
  column outside the frozen 121-column contract and rejects a frame missing
  any contract column — enforcing "never invent column names" structurally.
  Verified **byte-identical** output (all 121 rows, all columns including
  `reason` text) against the real fixture.
- `atlas/game_intelligence/pregame_team_identity_timeline.py`: rebuilt to
  compute a true strictly-prior-date expanding mean per team (doubleheader
  same-date games never see each other), with real column naming
  (`identity__expanding_mean__{source_column}`) and the real audit schema
  (`atlas_season, team, game_date, target_team_game_rows,
  expected_prior_games, observed_prior_games, prior_game_count_matches,
  same_date_games_used, future_games_used, representative_feature_checks,
  representative_feature_passes, all_feature_checks_pass, audit_pass`) and
  failures schema. Verified exact match (row-for-row, byte-identical
  `identity_games_before_date`/`identity_dates_before_date`/expanding-mean
  values, and audit fields) against the real 1,701-row timeline fixture and
  1,785-row audit fixture.
- `atlas/game_intelligence/pregame_identity_matchup_builder.py`: rebuilt with
  the real `identity_abs_edge__` naming, real interleaved column ordering
  (`team_identity__X, opponent_identity__X, identity_edge__X,
  identity_abs_edge__X` per feature), and all real summary-diagnostic
  columns. Verified exact 383-column ordering against the authoritative
  schema. **Confidence note**: no fixture row sample exists for the matchup
  artifact itself; `identity_sample_balance`,
  `identity_sample_confidence_score`, and
  `identity_sample_confidence_label` formulas are reconstructed from the
  schema's aggregate `numeric_profile`/`sample_values` statistics (moderate,
  not full-row confidence). If exact row-level verification becomes
  necessary, the minimum artifact required is a redacted parquet sample of
  `data/game_intelligence/pregame_identity_matchups/2024/
  pregame_identity_matchups.parquet` (or an equivalent
  `atlas_reference/samples/` addition) analogous to the registry/timeline
  samples already vendored.
- `atlas/config/paths.py`: `DATA_ROOT`/`CODE_ROOT` now honor
  `ATLAS_DATA_ROOT`/`ATLAS_CODE_ROOT` environment variable overrides, while
  defaulting to the original hard-coded
  `/content/drive/MyDrive/Project_Atlas/...` paths when unset. The
  production Google Drive workspace remains a fully supported default
  runtime option; no production entry point was changed.

**Tests added/changed**: all three test files
(`tests/test_pregame_identity_source_registry.py`,
`tests/test_pregame_team_identity_timeline.py`,
`tests/test_pregame_identity_matchup_builder.py`) were rewritten to test
against the real contract using `atlas_reference/samples/` fixtures instead
of self-referential synthetic data matching the old invented schema. Each
file keeps exactly one production-only integration test
(`test_2024_*_reproduction_against_production_workspace`) that explicitly
`pytest.skip()`s — never fails — when the full
`/content/drive/MyDrive/Project_Atlas` workspace is absent.

**Focused test results**: `python -m pytest
tests/test_pregame_identity_source_registry.py
tests/test_pregame_team_identity_timeline.py
tests/test_pregame_identity_matchup_builder.py -q` → **27 passed, 3
skipped** (the 3 skips are the production-only reproduction tests, correctly
skipped in this sandbox).

**Full regression suite**: `python -m pytest tests/ -q` → **106 passed, 111
failed, 3 skipped**. The 111 failures are unchanged from the pre-repair
baseline (96 passed/111 failed/2 skipped) and are entirely pre-existing
`FileNotFoundError`s for `/content/drive/MyDrive/Project_Atlas/...` paths
unrelated to this repair — no new failures were introduced, and the repair
added 10 net new passing tests (17 old fabricated-contract tests replaced by
27 real-contract tests, 3 of which newly skip cleanly instead of failing).

**Production validations still pending** (require the real Colab/Drive
workspace, cannot run in this sandbox):

- `test_2024_registry_reproduction_against_production_workspace`
- `test_2024_timeline_reproduction_against_production_workspace`
- `test_2024_matchup_reproduction_against_production_workspace`
- Full-season (4,856-row) 2024 regression of all three rebuilt artifacts
  against the production Drive parquet files.
- Row-level verification of the matchup summary-diagnostic formulas
  (`identity_sample_balance`, confidence score/label) against a real
  matchups fixture, once one is vendored.

**Repair commit**: recorded via `engine-tools-report_progress` immediately
following this ledger update (see git log for the exact hash).

## Post-merge verification (2026-07-18): Phase 2E.1–2E.3A repair

The recovery milestone above was merged to `main` via PR #2. This section
records an independent post-merge audit of that merged state — no code was
rebuilt.

**Merge commit**: `ff970bbbebde29fe2ae78fd2db62d9d259ad05c9` ("Merge pull
request #2 from brianrags07-afk/copilot/continue-phase-2e-development").

**Repair commit**: `08498e3a0263de8bbe1e291114ce3211a3b71e69` ("Repair
fabricated Phase 2E.1-2E.3A identity pipeline against real ATLAS contract"),
with a follow-up documentation-only commit `35636b4` ("Add clarifying
comment for identity sample thresholds") also included in the merge.

**Full regression suite**: `python -m pytest tests/ -q` → **106 passed, 111
failed, 3 skipped**. Identical to the counts recorded at repair time; all 111
failures are pre-existing `FileNotFoundError`/`AssertionError`s caused by the
absence of the `/content/drive/MyDrive/Project_Atlas` production workspace in
this sandbox (scoring timelines, team/game outcome classifiers, prediction
logic policy, etc.) — none are new and none touch the repaired identity
modules.

**Focused Phase 2E.1/2E.2/2E.3A suite**: `python -m pytest
tests/test_pregame_identity_source_registry.py
tests/test_pregame_team_identity_timeline.py
tests/test_pregame_identity_matchup_builder.py -v` → **27 passed, 3
skipped**, matching the repair-time result exactly.

**Contract conformance re-verified independently**:
- `atlas_reference/schemas/data__game_intelligence__pregame_identity_registry__2024__pregame_identity_source_registry.csv.schema.json`
  confirms the authoritative registry contract is 121 rows / 10 columns
  (`column, dtype, family, source_status, same_game_safe, requires_shift,
  historical_aggregation_allowed, non_null_rows, unique_values, reason`).
  `atlas/game_intelligence/pregame_identity_source_registry.py`'s
  `IDENTITY_SOURCE_CLASSIFICATION` table and `assert_matches_frozen_contract`
  enforce exactly this shape.
- `atlas/game_intelligence/pregame_team_identity_timeline.py` still computes
  a strictly-prior-date expanding aggregate
  (`_strictly_prior_date_expanding_aggregates`) — same-date (doubleheader)
  games are excluded from each other's history, preserving pregame/postgame
  temporal integrity.
- `atlas/config/paths.py` still defaults `DATA_ROOT`/`CODE_ROOT` to the
  original hard-coded `/content/drive/MyDrive/Project_Atlas` production
  paths, only overridable via `ATLAS_DATA_ROOT`/`ATLAS_CODE_ROOT` env vars;
  the production entry point was not altered.
- A repo-wide search for the previously fabricated column/naming patterns
  (`identity_feature_name`, `min_lagged_days`, `identity_edge_abs__`) returns
  zero matches in any `atlas/` or `tests/` file — only this ledger's
  historical description of the old defect still mentions them.
- No duplicate identity-pipeline implementations exist: exactly one module
  each for the registry, timeline, and matchup builder, with no other
  `atlas/` file importing or reimplementing their logic.
- One pre-existing, unrelated dead-code item was noted at the time (not
  part of that repair, out of scope then): `atlas/config.py` was a legacy
  module shadowed by the `atlas/config/` package (which all identity
  modules and tests correctly import from); it predated that repair
  (`git log` showed its last real change in the original "Daily Data
  Engine v1" commit). That prior note incorrectly concluded it was "not
  read by any current code path" — in fact `atlas/gamecards/
  gamecard_engine.py` and `atlas/daily/data_engine.py` imported
  `GAMECARD_DIR`, `MLB_API`, and `today_str` from `atlas.config`, names
  that existed only in the dead shadowed file, not in the `atlas/config/`
  package. Because the package always wins the import (confirmed via
  `atlas.config.__file__`), both modules raised `ImportError` at import
  time. This was fixed in the dev-data-bundle session below: the missing
  names were added to `atlas/config/paths.py`/`atlas/config/__init__.py`
  and the dead `atlas/config.py` file was deleted.

**Skipped production-only tests and required Google Drive artifacts** (all
three skip cleanly, never fail, when these are absent — as confirmed in this
sandbox):

1. `tests/test_pregame_identity_source_registry.py::test_2024_registry_reproduction_against_production_workspace`
   requires `$ATLAS_DATA_ROOT/game_intelligence/pregame_identity_registry/2024/pregame_identity_source_registry.csv`
   (default: `/content/drive/MyDrive/Project_Atlas/data/game_intelligence/pregame_identity_registry/2024/pregame_identity_source_registry.csv`).
2. `tests/test_pregame_team_identity_timeline.py::test_2024_timeline_reproduction_against_production_workspace`
   requires all three of:
   `.../game_intelligence/game_flow_facts/2024/team_game_flow_facts.parquet`,
   `.../game_intelligence/pregame_identity_registry/2024/pregame_identity_source_registry.csv`,
   `.../game_intelligence/pregame_team_identities/2024/pregame_team_identity_timeline.parquet`.
3. `tests/test_pregame_identity_matchup_builder.py::test_2024_matchup_reproduction_against_production_workspace`
   requires all three of:
   `.../game_intelligence/pregame_team_identities/2024/pregame_team_identity_timeline.parquet`,
   `.../game_intelligence/pregame_identity_registry/2024/pregame_identity_source_registry.csv`,
   `.../game_intelligence/pregame_identity_matchups/2024/pregame_identity_matchups.parquet`.

All paths above are rooted at `$ATLAS_DATA_ROOT` (default
`/content/drive/MyDrive/Project_Atlas/data`), i.e. the Google
Drive-backed Colab production workspace.

**Verdict**: the merged repair is clean. No schema drift, no invented
columns, no duplicate implementations, no new hard-coded paths, no temporal
leakage, and no broken callers were found. No corrective PR is needed.

**Next milestone**: Phase 2E.5A (2025 blind-validation input readiness)
remains the next authoritative unfinished ATLAS milestone, and remains
blocked by **STOP CONDITION 3** — the real 2025 pregame identity, bullpen,
and lineup-starter production artifacts exist only in the Google
Drive-backed Colab runtime and are not present in this sandbox or vendored
into `atlas_reference/samples/`. Per governance, no new builder or test may
be written against fabricated 2025 data. This verification session does not
change that blocker; work on 2E.5A can resume once production Drive access
or a governed redacted 2025 sample is available.

## Data-bridge tooling (2026-07-18): versioned GitHub Release dev-data bundle

This session did **not** advance Phase 2E.5A's scientific work (still
blocked by STOP CONDITION 3). Instead, per an explicit request, it built the
safe infrastructure to remove the hard Google Drive dependency for future
sessions/developers, without fabricating or uploading any data:

- `atlas_reference/dev_data_bundle_required_artifacts.json` — the minimum
  real production artifacts required to execute Phase 2E.5A through Phase
  2E completion, in dependency order. Every `known_*` value in it is copied
  verbatim from the already-committed `atlas_reference/schemas/` catalog
  (a real prior production scan); artifacts whose 2025 path could not be
  confirmed against that catalog (2025 identity timeline/matchups, the
  lineup-starter interaction output — two candidate paths exist across
  this repo's own docs and are both listed) are explicitly marked
  `path_unconfirmed_requires_colab_verification` with no fabricated
  `known_*` values.
- `atlas_reference/manifests/dev_data_bundle_manifest.schema.json` — the
  manifest specification (original/bundled path, size, row/column count,
  primary key, SHA-256, season, purpose).
- `scripts/dev_data_bundle/colab_package_dev_data_bundle.py` — Colab
  packaging script: reads only the allowlisted artifacts from
  `/content/drive/MyDrive/Project_Atlas`, verifies existence/grain
  uniqueness, stages, writes the manifest, compresses, and splits into
  <2 GiB parts.
- `scripts/dev_data_bundle/bootstrap_dev_data_bundle.py` — downloads a
  private GitHub Release asset via authenticated GitHub API access,
  verifies every SHA-256 (parts, reassembled archive, extracted
  artifacts), extracts outside the repository, and sets/documents
  `ATLAS_DATA_ROOT`. Fails with a distinct exit code for each of:
  missing/invalid auth, missing release/asset, checksum mismatch, and
  missing files after extraction.
- `atlas/config/paths.py` / `atlas/config/__init__.py`: added the
  previously-missing `GAMECARD_DIR`/`MLB_API`/`today_str`/`RAW_DIR`/
  `DAILY_DIR`/`SNAPSHOT_DIR`/`ensure_dirs` names (see the corrected
  dead-code note above) so every production module that imports from
  `atlas.config` honors `ATLAS_DATA_ROOT`; deleted the now-fully-superseded
  dead `atlas/config.py`. The Google Drive path remains the untouched
  default when `ATLAS_DATA_ROOT` is unset.
- `docs/DEV_DATA_BUNDLE.md` — the packaging/upload/bootstrap commands and
  the bundle versioning/replacement policy.
- Tests: `tests/test_dev_data_bundle_manifest.py`,
  `tests/test_dev_data_bundle_packaging.py`,
  `tests/test_dev_data_bundle_bootstrap.py`,
  `tests/test_atlas_config_paths.py` — 48 new tests, all against synthetic
  fixtures / monkeypatched network calls, none touching real production
  data.

**Full regression suite**: `python -m pytest tests/ -q` → **154 passed
(106 → 154), 111 failed, 3 skipped**. The 111 failures and 3 skips are
unchanged and are the same pre-existing Drive-absence cases documented
above; none are new.

**Effect on STOP CONDITION 3**: unchanged. This tooling does not itself
retrieve or vendor any 2025 production data — a maintainer with real Colab
Drive access must still run the packaging script and publish a release
before the bootstrap script has anything to download. Once that happens,
Phase 2E.5A can proceed using `ATLAS_DATA_ROOT` instead of a live Drive
mount.


