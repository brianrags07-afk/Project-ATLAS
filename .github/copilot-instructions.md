# Project ATLAS Copilot Instructions

Before modifying an ATLAS data builder:

1. Read `atlas_reference/manifest.json`.
2. Inspect the applicable schema under `atlas_reference/schemas/`.
3. Check authoritative registries under `atlas_reference/registries/`.
4. Use `atlas_reference/manifests/relationship_map.json`.
5. Test against applicable samples under `atlas_reference/samples/`.
6. Never invent column names.
7. Never use player names as durable keys when player IDs exist.
8. Never silently rename incompatible production columns.
9. Preserve strict pregame/postgame temporal integrity.
10. Frozen production artifacts are contracts and must not be changed merely to satisfy new code.
