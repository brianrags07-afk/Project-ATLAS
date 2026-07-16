# ATLAS Brain Phase 2E.4F.3 — Feature Semantic Governance

## Final repaired status

Engine version: 1.0.1

The initial semantic audit correctly blocked entity identifiers and direct
target analogues, but classified recent bullpen usage as a generic exposure
proxy.

That classification was repaired.

## Workload distinction

`bullpen_games_used_prior_3_dates` measures recent bullpen demand before the
current game. It is not a historical sample-size field.

It is now classified as:

- `VALID_RECENT_WORKLOAD_FACT`
- `KEEP_SEMANTICALLY_VALID`

Historical career and season opportunity fields remain reviewable exposure
proxies.

## Final scope

- Phase 2E.4F.2 representatives preserved: 4,271
- Concept member rows audited: 8,542
- Semantically valid freeze candidates: 2,138
- Review-required concepts: 0
- Blocked concepts: 1,089
- Transformation-family redundant concepts:
  1,044

## Safety

- Identifier thresholds remain blocked.
- Target analogues remain blocked.
- No concept rows were deleted.
- No weights were assigned.
- No predictions were created.
- No 2025 validation results were used.
- No 2026 results were used.

## Next phase

Create the immutable 2024 concept-definition freeze registry and content
fingerprints before blind 2025 validation.
