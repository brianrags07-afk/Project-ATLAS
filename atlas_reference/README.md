# ATLAS Development Data and Contract Pack

This directory is generated from the full Project ATLAS production data stored in Google Drive.

## Purpose

The pack gives GitHub Copilot, Codex, CI, and developers access to the real ATLAS data structures
without committing the complete production history.

## Contents

- `schemas/`: Actual columns, dtypes, null profiles, candidate identifiers, and sample values.
- `samples/general/`: Representative samples from all discovered tabular artifacts.
- `samples/players/`: Samples emphasizing player-linked rows.
- `samples/games/`: Samples emphasizing complete game-linked structures.
- `registries/`: Small registries, mappings, dictionaries, and contract-like artifacts copied intact.
- `metadata/`: Small JSON, YAML, and TOML metadata files.
- `manifests/`: Table catalogs, player catalogs, relationship candidates, samples, and checksums.
- `reports/`: Build errors or warnings.

## Rules

1. Do not invent columns that are not present in these schemas.
2. Do not use player names as durable keys when player IDs exist.
3. Do not classify a field as pregame-safe merely because it appears in a sample.
4. Authoritative registries override heuristic classifications.
5. Production builders must be validated against frozen expected artifacts.
6. Samples are for engineering correctness; full Drive data remains required for scale validation and training.
7. Schema incompatibilities must fail explicitly rather than being silently renamed or coerced.
