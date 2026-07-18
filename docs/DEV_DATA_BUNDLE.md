# ATLAS Development Data Bundle

This document describes the versioned, checksum-verified GitHub Release
data-bridge that replaces direct Google Drive dependency for local/CI
development against real ATLAS production artifacts.

It has three parts:

1. **Packaging** (run in Colab, with Drive mounted) -- builds a bundle
   containing only the real production files listed in
   `atlas_reference/dev_data_bundle_required_artifacts.json`.
2. **Release upload** (run by a maintainer with push access) -- publishes
   the bundle as a **private** GitHub Release asset.
3. **Bootstrap** (run on a developer machine, CI runner, or by an agent with
   `runtime-tools` / authenticated `gh` access) -- downloads, verifies, and
   extracts the bundle outside the repository, and points
   `ATLAS_DATA_ROOT` at it.

No step here fabricates data. The packaging script only ever copies files
that already exist under the Drive project root; if a required artifact is
missing, packaging fails with a clear error instead of substituting a
placeholder.

## 1. Required artifacts

`atlas_reference/dev_data_bundle_required_artifacts.json` is the allowlist
of real production files needed to execute Phase 2E.5A through Phase 2E
completion, in dependency order. Each entry records:

- `original_production_path` -- path under `/content/drive/MyDrive/Project_Atlas`
- `season`, `purpose`, `primary_key`, `consuming_builder`
- `catalog_status` -- `confirmed_in_schema_catalog` when the artifact's
  real size/row-count/column-count/SHA-256 are already recorded in
  `atlas_reference/schemas/` (from a prior real production scan), or
  `path_unconfirmed_requires_colab_verification` when the artifact's exact
  2025 path/existence could not be confirmed from repository evidence alone.

The packaging script always re-measures every value directly from the live
file; the registry's `known_*` fields are provenance/cross-check hints
only, never a substitute for a fresh measurement.

## 2. Packaging (in Colab)

```bash
# Run inside a Colab notebook cell, with Drive mounted at
# /content/drive/MyDrive/Project_Atlas.
!python scripts/dev_data_bundle/colab_package_dev_data_bundle.py \
    --project-root /content/drive/MyDrive/Project_Atlas \
    --output-dir /content/atlas_dev_data_bundle_build \
    --bundle-version 1.0.0
```

This produces, under `--output-dir`:

- `staging/` -- the staged copy of every required artifact (for
  inspection before upload; safe to delete afterward)
- one of:
  - `atlas-dev-data-bundle-1.0.0.tar.gz` (if under the 2 GiB part-size
    ceiling), or
  - `atlas-dev-data-bundle-1.0.0.tar.gz.part000`, `.part001`, ... (if the
    archive was split)
- `release_manifest.json` -- conforms to
  `atlas_reference/manifests/dev_data_bundle_manifest.schema.json`; lists
  every artifact's real measured size, row count, column count, primary
  key, and SHA-256, plus the archive/part checksums needed for
  verification.

The script fails (non-zero exit, clear message) if:

- the Drive project root does not exist,
- a required artifact cannot be found at any of its candidate paths,
- an artifact's declared primary-key columns are missing, or
- an artifact is not unique at its declared primary-key grain.

## 3. GitHub Release upload

Upload the archive (or all parts) and `release_manifest.json` as assets on
a **private** release of this repository. Using the GitHub CLI:

```bash
gh release create v1.0.0 \
    --repo brianrags07-afk/Project-ATLAS \
    --title "ATLAS dev data bundle v1.0.0" \
    --notes "See docs/DEV_DATA_BUNDLE.md" \
    /content/atlas_dev_data_bundle_build/release_manifest.json \
    /content/atlas_dev_data_bundle_build/atlas-dev-data-bundle-1.0.0.tar.gz*
```

(The trailing `*` picks up either the single archive or all its
`.partNNN` files, whichever exist.) The repository must remain private, or
the release must be marked private/draft, so the bundle is never publicly
downloadable without authentication.

## 4. Bootstrap (downloading and verifying)

On a developer machine, CI runner, or an agent with authenticated GitHub
access (never inside a sandboxed agent with no network path to GitHub or
Drive):

```bash
export GITHUB_TOKEN=ghp_xxx   # or GH_TOKEN; must have read access to the private release
python scripts/dev_data_bundle/bootstrap_dev_data_bundle.py \
    --repo brianrags07-afk/Project-ATLAS \
    --tag v1.0.0 \
    --dest ~/atlas_dev_data
```

This will:

1. Fetch the release and locate `release_manifest.json` and the archive
   (or all parts) via authenticated GitHub API requests.
2. Verify every downloaded part's SHA-256, then the reassembled archive's
   SHA-256, against the manifest.
3. Extract the archive **outside the repository** (default
   `~/atlas_dev_data/<bundle_version>/`).
4. Verify every individual artifact's SHA-256 after extraction.
5. Set `ATLAS_DATA_ROOT` for the current process and write
   `~/atlas_dev_data/atlas_dev_data.env`, which can be sourced to persist
   it:

   ```bash
   source ~/atlas_dev_data/atlas_dev_data.env
   ```

The script fails clearly and distinctly (see its module docstring for the
full list) for missing/invalid authentication, a missing release/asset, a
checksum mismatch, or files missing after extraction. It never falls back
to fabricating a file.

## 5. `ATLAS_DATA_ROOT` and the Google Drive default

`atlas/config/paths.py` resolves `DATA_ROOT` from the `ATLAS_DATA_ROOT`
environment variable, defaulting to the original hard-coded
`/content/drive/MyDrive/Project_Atlas/data` Google Drive path when unset.
Setting `ATLAS_DATA_ROOT` (e.g. via the sourced `.env` file above) makes
every production module that imports `DATA_DIR`/`DATA_ROOT`/`MASTER_DIR`/
etc. from `atlas.config` resolve against the bootstrapped bundle instead,
with no code changes required. The Google Drive path is never removed as
the default; both runtimes remain fully supported.

## 6. Bundle versioning and replacement policy

- Bundle versions are semantic (`MAJOR.MINOR.PATCH`) and correspond
  1:1 with a GitHub Release tag (`v<version>`).
- **PATCH**: re-packaging the same artifact set after a production
  artifact was corrected/regenerated upstream (row counts/checksums may
  change; artifact set does not).
- **MINOR**: adding a newly-required artifact to
  `atlas_reference/dev_data_bundle_required_artifacts.json` (e.g. once a
  2025 identity artifact's real path is confirmed and no longer
  `path_unconfirmed_requires_colab_verification`).
- **MAJOR**: removing or restructuring artifacts, or changing the bundle's
  manifest schema in a way that breaks older bootstrap script versions.
- Bundles are immutable once released: never edit or re-upload assets
  under an existing tag. Publish a new tag instead, even for a one-file
  fix.
- Old bundle versions are not deleted automatically; a maintainer may
  mark a release "deprecated" in its notes once no active branch depends
  on it, but the tag and assets remain available for reproducibility.
- `docs/AUTOPILOT_EXECUTION_LEDGER.md` records which bundle version (if
  any) was used to validate a given phase.
