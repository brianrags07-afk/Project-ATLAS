# Project Atlas Changelog

> **Status note:** This log predates Phases 2A–2E. See the
> `docs/ATLAS_BRAIN_PHASE_*.md` series and
> `docs/AUTOPILOT_EXECUTION_LEDGER.md` for phase-by-phase history and the
> current authoritative checkpoint.

## Unreleased

- **Recovery**: Rebuilt Phase 2E.1 (`pregame_identity_source_registry.py`),
  2E.2 (`pregame_team_identity_timeline.py`), and 2E.3A
  (`pregame_identity_matchup_builder.py`), which had been implemented
  without access to the real ATLAS schemas and shipped fabricated column
  names/semantics. Rebuilt against the authoritative contract pack under
  `atlas_reference/`; verified byte-identical output against real fixtures
  for the registry and timeline. See
  `docs/AUTOPILOT_EXECUTION_LEDGER.md` for full root-cause analysis and
  evidence.
- `atlas/config/paths.py`: `DATA_ROOT`/`CODE_ROOT` now support
  `ATLAS_DATA_ROOT`/`ATLAS_CODE_ROOT` environment variable overrides while
  preserving the production Google Drive path as the default.

## v0.1.0

Created: 2026-07-04

- Initialized Project Atlas
- Created folder structure
- Created Project Charter
- Started core documentation
