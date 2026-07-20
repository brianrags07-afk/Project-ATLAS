# ATLAS Known Issues / Follow-Up Work

## OPEN-1: `bullpen_concept_validation_2025.run_bullpen_concept_validation_2025`
is broken by the lineage-complete rewrite of `concept_validation_2025`

**Status:** Open. Tracked here as a documented follow-up; not fixed in the
concept-validation lineage-completeness change (see
`atlas/validation/concept_validation_2025.py`).

**Root cause:**

`atlas/validation/bullpen_concept_validation_2025.py` reuses the 2025
concept-validation engine internally by monkey-patching module-level
globals on `atlas.validation.concept_validation_2025` (see
`_redirect_validation_globals` / `_restore_validation_globals`) and by
calling `run_concept_validation_2025(only_team=..., limit=..., resume=...)`.

The lineage-complete rewrite of `concept_validation_2025`:

- removed the `CONCEPT_REGISTRY_PATH` / `CONCEPT_MEMBER_MAP_PATH` globals
  that `_redirect_validation_globals` expects to patch (replaced by
  `FROZEN_DEFINITION_REGISTRY_PATH` / `FROZEN_MEMBER_REGISTRY_PATH`), and
- changed `run_concept_validation_2025()` to take no arguments (team
  scoping, `limit`, and `resume` no longer exist in the lineage-complete
  engine).

As a result, calling
`bullpen_concept_validation_2025.run_bullpen_concept_validation_2025()`
now raises `AttributeError: Existing validation engine is missing
expected global: CONCEPT_REGISTRY_PATH` before it ever reaches
`run_concept_validation_2025`.

**Blast radius:** Nothing in the current test suite or any other module
imports or calls `run_bullpen_concept_validation_2025`, so this break is
currently isolated to that one function. Importing the module itself
remains completely safe (see
`tests/test_concept_validation_2025.py::test_bullpen_validation_module_imports_cleanly`),
so no other code path is affected today.

**Follow-up:** Rewrite
`atlas/validation/bullpen_concept_validation_2025.py` to build its own
lineage-complete bullpen validation pass directly against the frozen
bullpen concept registries (mirroring the approach in
`concept_validation_2025.py`) instead of monkey-patching the 2025 concept
engine's internals. This should be scoped as its own change so it can be
reviewed against `atlas_reference/schemas` and
`atlas_reference/registries` for the bullpen concept artifacts, rather
than folded into the concept-validation lineage-completeness change.
