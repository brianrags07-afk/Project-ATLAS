# ATLAS Historical Readiness Audit

This document describes the redesigned ATLAS historical readiness audit
tooling under `atlas/audit/`, its report contracts, and the safety
guarantees enforced by its workflows and tests.

The audit is **strictly read-only** with respect to Cloud Storage. It
never writes to, deletes from, renames, moves, or overwrites any object
in `gs://atlas-mlb-data-brian-4817`. It never runs a 2024 rebuild, a 2025
walk-forward backtest, model training, or prediction generation. It only
produces reports, written to `artifacts/audits/` for upload as GitHub
Actions artifacts.

## Why a single "coverage status" was replaced

The previous design collapsed several fundamentally different evidence
questions into one status field per dataset/season. That made it
possible to accidentally treat "the data exists" as "the data is safe
to use for pregame prediction" -- exactly the kind of confusion that
causes look-ahead leakage. The redesign makes each question its own,
independently computed field.

## The five independent evidence dimensions

Every coverage-matrix row/season now reports all five of the following,
computed independently unless an explicit, documented, tested rule
says otherwise:

| Dimension | Values | Question it answers |
|---|---|---|
| `data_presence` | `present`, `partial`, `missing`, `unknown` | Does *any* copy of this data exist anywhere we looked? |
| `source_completeness` | `complete`, `partial`, `incomplete`, `unknown`, `not_applicable` | Is the data we found complete for the season/scope in question? |
| `provenance_status` | `verified`, `partial`, `missing`, `unknown` | Do we know exactly where this data came from (hashes, manifests, lineage)? |
| `temporal_availability` | `pregame_proven`, `postgame_only`, `mixed`, `unknown`, `not_applicable` | Do we have timestamp evidence that this value was knowable *before* the relevant game's cutoff? |
| `pregame_safety` | `safe`, `unsafe`, `conditional`, `unknown`, `not_applicable` | Is it safe to use this value as a same-game pregame prediction input? |

Only two functions are allowed to derive one dimension from another,
and both are explicit, documented, and unit tested:

- `atlas.audit.temporal_proof.assess_pregame_safety_from_temporal_availability`
  (`temporal_availability` -> `pregame_safety`).
- `atlas.audit.schedule_source_assessment.assess_schedule_source`, which
  sets `provenance_status`, `temporal_availability`, and `pregame_safety`
  together for schedule/series-context evidence, per the published-schedule
  rule below.

Every other row is computed independently: a dataset can be
`source_completeness=complete` and simultaneously `pregame_safety=unsafe`
(see "complete but unsafe" below), and `data_presence=present` never by
itself implies anything about provenance or temporal safety.

Each row/season also carries:

- `evidence`: a list of structured evidence records (see contract below).
- `risks`: plain-language leakage/incompleteness risks specific to that row.
- `required_next_evidence`: what would need to be collected to move the
  dimensions forward (e.g. "a timestamped published-schedule source").

## Data layers

`atlas.audit.dataset_profile.classify_data_layer` distinguishes:

- `raw_source` -- original, unprocessed source objects.
- `normalized_master` -- master/normalized datasets such as
  `master_game_database.parquet`, `master_pitch_database.parquet`, and
  `team_game_state.parquet`.
- `derived_feature` -- feature tables computed from normalized data.
- `learned_artifact` -- fitted models, concept definitions, learned
  parameters.
- `prediction_artifact` -- generated predictions/Game Cards.
- `report_or_manifest` -- audit reports, manifests, schemas.

Classification is **heuristic** (based on path/name patterns) and is
always tagged with `data_layer_confidence in {"heuristic", "unknown"}` --
it is never promoted to `"verified"` or `"observed"`. Critically,
`master_game_database.parquet` and `master_pitch_database.parquet` are
always classified `normalized_master`, and their presence is **never**
treated as proof that the original raw source objects exist or are
complete -- raw-source readiness must be assessed against actual raw
objects, not inferred from the master tables that were built from them.

## The published-schedule rule (no-leakage)

`published_schedule` and `published_series_context` are two of the most
leakage-prone facts in the system because they are trivially "complete"
after the fact (every game that happened is, retroactively, "the
schedule"). The audit enforces:

- Completed-game records in `master_game_database` can prove
  `data_presence` for *game identifiers*, but they **never** promote
  `published_schedule`'s `provenance_status` to `verified` or its
  `pregame_safety` to `safe`. Evidence of this kind is tagged
  `evidence_type="completed_game_record"` and always yields
  `provenance_status in {"missing", "unknown"}` /
  `pregame_safety in {"unsafe", "unknown"}`.
- Only a genuinely timestamped published-schedule source (evidence type
  `published_schedule_source`, with a retrieval/publish timestamp
  provably before the games it describes) can yield
  `provenance_status="verified"` and `pregame_safety="safe"`.
- Series length/boundaries **inferred from completed game history**
  (evidence type `series_inferred_from_results`) are explicitly
  `pregame_safety="unsafe"` -- they must never authorize pregame
  prediction, even though the inferred series length is "correct" in
  hindsight.

See `atlas/audit/schedule_source_assessment.py` for the implementation
and `tests/test_atlas_audit_schedule_source_assessment.py` for the three
required test cases.

## Complete-but-unsafe: postgame facts

Final scores, pitch-by-pitch data, plate appearances, and batted-ball
events are postgame facts. The audit allows these to be
`source_completeness="complete"` (they are valid, complete raw inputs
for *historical reconstruction and learning*) while simultaneously
`temporal_availability="postgame_only"` and `pregame_safety="unsafe"`
(or `not_applicable` when absent). This is the canonical example of why
the dimensions must stay independent: collapsing them into one status
would either falsely mark postgame facts as usable pregame inputs, or
falsely mark valid reconstruction inputs as "not ready" for anything.

## Dynamic pregame fields require per-game timestamp proof

Starters, lineups, bullpen availability, injuries, weather, umpires, and
market information are all dynamic and only pregame-safe if we have
evidence that we *knew* the value before that specific game's
`feature_cutoff_time`. `atlas.audit.temporal_proof` implements this:

- `has_per_game_timestamp_proof(...)` checks for a `source_retrieved_at`
  (or equivalent) timestamp paired to a specific game, strictly before
  that game's cutoff.
- Without that per-game pairing, `assess_field_temporal_availability`
  returns `unknown` (never `unsafe` by default, and never `safe`).
- Object **upload timestamps** in Cloud Storage are explicitly *not*
  accepted as proof of original real-world availability. Storage timing
  is reported separately (`storage_timestamp_note`) and is never used to
  promote `temporal_availability` or `pregame_safety`.

## Readiness decisions A-G

`atlas.audit.readiness.build_readiness_decisions` returns one record per
decision. Each decision independently selects the evidence dimensions
and thresholds it requires -- there is no single generic function that
treats every non-"complete" state the same way. Every decision returns:

- `verdict`
- `required_dimensions` (and the thresholds evaluated against them)
- `evidence_used`
- `blockers`
- `warnings`
- `next_action` (the exact next step to move the decision forward)
- `does_not_authorize` -- what this verdict does **not** permit, always
  including the baseline statement that no verdict authorizes a 2024
  rebuild, 2025 backtest, model training, or prediction generation by
  itself.

| Decision | What it requires | What "ready" would authorize |
|---|---|---|
| A -- Exact 2024 reproduction | Source hashes, code/version lineage, manifests, artifact lineage, transformation identity, schema versions, reproducible environment evidence. Processed-table presence alone is insufficient. | Re-running the exact, already-validated 2024 pipeline bit-for-bit. |
| B -- Rebuild 2024 from raw | Raw-source presence/completeness, provenance and hashes, immutable/versioned source evidence, transformation availability. The postgame nature of raw pitch/game facts is *not* a blocker. | A staging-only rebuild of 2024 from raw sources, never overwriting production outputs. |
| C -- Freeze 2024 learned artifacts | Artifact provenance, manifest ID, source hashes, code commit, schema version, validation status, immutable destination/versioning. | Treating a specific artifact version as frozen/citable. |
| D -- Parse 2025 with identical transformations | Schema compatibility checks, transformation/version identity, an explicit compatibility report, no silent renaming/coercion. | Applying the *same* transformation code to 2025 raw inputs. |
| E -- 2025 chronological walk-forward backtest | Timestamp-proven pregame features at every game cutoff, strict chronological state updates, predictions frozen before outcomes, leakage tests. Final full-season tables cannot authorize this. | Running an actual walk-forward backtest (this audit itself still never runs one). |
| F -- 2025 historical Pregame Game Cards | Field-level temporal provenance, preserved null/unknown values, no postgame backfill. | Generating historical Game Cards for study/analysis. |
| G -- 2026 forward predictions | Published schedule, current pregame snapshots, frozen model/artifact versions, no same-game outcomes, complete run manifesting. | Generating a real forward prediction run. |

**What no verdict automatically authorizes:** a "ready" verdict on any
of A-G is evidence-sufficiency information only. It does not itself
run a rebuild, a backtest, training, or prediction, and it does not
waive any of the other decisions' independent requirements.

## Structured evidence and provenance contracts

- `schemas/evidence_record.schema.json` defines the shape of every
  evidence record: `evidence_type`, `source`, `path_or_object`,
  `field_or_column`, `season`, `observed_value`, `confidence`,
  `limitation`.
- `schemas/historical_readiness_report.schema.json` defines the shape of
  the full coverage-matrix + readiness report and `$ref`s the evidence
  record schema.
- `atlas.audit.report_schema.validate_report(matrix, readiness)` validates
  a generated report against these schemas before it is written; a
  schema-invalid report is a hard failure of the audit run.
- `atlas.audit.provenance.build_dataset_provenance` merges Cloud Storage
  object metadata (generation, metageneration, MD5/CRC32C, size,
  created/updated timestamps) with dataset-profile fields (schema
  fingerprint, row count, season/date range, candidate primary key,
  duplicate-key count) into one record per dataset, reaching
  `provenance_status="verified"` only when **both** a matched content
  hash **and** an explicit manifest linkage are present. Missing
  evidence is left `missing` or `unknown` and is never inferred.

## Report format / filename migration notes

- `historical_coverage_matrix.csv` and `.md` are still written, but their
  columns have changed: the old single `status` column is replaced by
  the five dimension columns (`data_presence`, `source_completeness`,
  `provenance_status`, `temporal_availability`, `pregame_safety`) plus
  `risks` and `required_next_evidence`.
- A new `historical_coverage_matrix.json` file is now also written,
  containing the full structured evidence records per row (the CSV/MD
  formats summarize evidence for readability; the JSON file is the
  complete, schema-validated record).
- `historical_readiness_decisions.*` now uses `verdict` instead of the
  old `decision` key, and adds `required_dimensions`, `evidence_used`,
  `blockers`, `warnings`, and `does_not_authorize` to every decision.

## Running the manual real-bucket audit safely

`.github/workflows/atlas-historical-readiness-audit.yml` remains
`workflow_dispatch`-only. `scripts/run_historical_readiness_audit.py`
is hardened so that:

- A **missing** expected master file (e.g. `team_game_state.parquet`
  not found in the bucket) is recorded as missing evidence and the
  audit continues, producing `unknown`/`missing` findings for that
  dataset -- it does not crash the job.
- The job fails **only** for: an authentication/authorization failure
  while listing or downloading from the bucket, a corrupt/unreadable
  input that was successfully downloaded but cannot be parsed, an
  output report that fails JSON Schema validation, or a violated safety
  invariant.
- No rebuild, backtest, training, or prediction step is ever invoked by
  this script.

## Continuous integration (no GCP access)

`.github/workflows/atlas-audit-ci.yml` runs on every pull request and
push (scoped to the audit-relevant paths), using Python 3.12, and:

- runs the full `tests/test_atlas_audit_*.py` suite,
- performs syntax/import validation of every `atlas/audit/*.py` module
  and the audit script,
- runs the static read-only guard tests
  (`tests/test_atlas_audit_workflow_no_cloud_writes.py`) and a
  self-check that the CI workflow itself never authenticates to GCP or
  references the real bucket.

It never authenticates to GCP and never accesses the real bucket. The
only workflow that ever touches the real bucket is the manual
`atlas-historical-readiness-audit.yml` workflow described above.

## Test coverage

The audit and schema test suite (`tests/test_atlas_audit_*.py`) proves,
among other things:

- Independent dimensions never overwrite each other (e.g. a row can be
  `source_completeness=complete` and `pregame_safety=unsafe`
  simultaneously).
- Processed master tables never satisfy raw-source readiness checks.
- Completed-game tables never satisfy published-schedule provenance.
- Series length inferred from results is always `pregame_safety=unsafe`.
- Missing per-game timestamps leave decision E blocked/unknown, never
  falsely "ready".
- Timestamp-proven snapshots can satisfy decision F's requirements.
- Decision A (exact reproduction) stays blocked without manifests and
  source hashes.
- Decision B (rebuild from raw) can be ready with postgame raw facts
  when raw provenance/completeness are independently verified.
- Schema incompatibility blocks decision D.
- The leakage guard in decision E rejects same-game outcome-derived
  inputs.
- Unknown evidence stays unknown rather than being coerced to a default.
- Generated reports validate against the JSON Schema contracts.
- No Cloud Storage mutation command exists in any workflow or in the
  audit script.
