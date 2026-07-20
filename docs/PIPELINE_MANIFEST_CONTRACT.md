# Pipeline Manifest Contract

This document defines the contract for the ATLAS **Pipeline Manifest**,
whose JSON Schema lives at
[`schemas/pipeline_manifest.schema.json`](../schemas/pipeline_manifest.schema.json).

A pipeline manifest is the append-only record of a single pipeline run
(discovery, backtest, forward prediction, rebuild, or audit). Every
builder/backtest/rebuild run must produce one before any output is
promoted out of staging, and a
[Pregame Game Card](PREGAME_GAME_CARD_CONTRACT.md)'s
`temporal_provenance.source_manifest_id` must reference one of these.

## Primary key

`pipeline_run_id` (globally unique).

## Fields

* `run_mode`: one of `discovery_2024`, `backtest_2025`, `forward_2026`,
  `rebuild`, `audit`.
* `season`: the season this run pertains to.
* `commit_sha`: the exact code commit that produced this run.
* `pipeline_version`: the semantic/build version of the pipeline code.
* `source_objects`: array of `{path, hash, hash_algorithm, date_range,
  season_range, row_count, column_count, schema_ref}` -- every input this
  run read, with a content hash so later audits can prove the input was
  unchanged.
* `output_objects`: array of `{path, hash, hash_algorithm, row_count,
  column_count, schema_ref, promotion_status}` -- every artifact this run
  produced, and whether it has been promoted out of staging.
* `started_at_utc` / `completed_at_utc`: run timestamps.
* `last_completed_game`: the last `{game_pk, game_date}` this run
  processed, to support safe incremental resumption.
* `run_extent`: `full` or `incremental`.
* `rerun_reason`: free-text reason if this run supersedes a prior one.
* `validation_results` / `leakage_audit_results`: `{status: passed |
  failed | not_run | unknown, details}`.
* `promotion_status`: `staging`, `promoted`, or `rejected`.
* `error_status`: `ok`, `failed`, or `unknown`.
* `parent_manifest_id`: the manifest this run was rerun from, if any.
* `frozen_discovery_artifact_ids` / `model_artifact_ids`: links to the
  concept-discovery and model artifacts this run froze or consumed.

## Rules

1. **Append-only.** A manifest is never edited in place. A rerun produces
   a new manifest with `parent_manifest_id` pointing at the prior one.
2. **No promotion without a passing audit.** `promotion_status` may only
   become `promoted` once `validation_results.status == "passed"` and
   `leakage_audit_results.status == "passed"`.
3. **Hashes are mandatory for both source and output objects.** A manifest
   with a missing hash cannot be used to prove reproducibility and must be
   treated as `error_status: "failed"`.
4. **All future generated outputs go to staging first.** `output_objects`
   entries start at `promotion_status: "staging"`; promotion is a
   separate, explicit, and auditable step -- never an implicit side effect
   of a run completing.
5. **No future game may influence an earlier prediction.** For
   `run_mode: backtest_2025` and `run_mode: forward_2026`, every
   `source_objects[].date_range.max_date` referenced while producing a
   given game's output must be strictly less than that game's own
   `game_date` (schedule fields excepted, per the Pregame Game Card
   contract's leakage prevention rules).
