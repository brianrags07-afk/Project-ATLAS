# Project Atlas Roadmap

Version: 0.1.0

> **Status note:** This document describes the original Sprint-1-era
> planning roadmap and is retained for historical context. It predates
> Phases 2A–2E. For the authoritative, verified current milestone, see
> `docs/AUTOPILOT_EXECUTION_LEDGER.md` and the `docs/ATLAS_BRAIN_PHASE_*.md`
> series. As of this update, Phases 1, 2A, 2B, 2C, 2D, and Phase 2E through
> 2E.4G (immutable 2024 concept-definition freeze) are frozen; Phase 2E.5A
> (2025 blind-validation input readiness) is the current open phase.

## Sprint 1 — Foundation
- [x] Create Project Atlas folder structure
- [x] Create Project Charter
- [ ] Create configuration file
- [ ] Create manifest file
- [ ] Create build report

## Sprint 2 — Statcast Import
- [ ] Import 2024 Statcast
- [ ] Import 2025 Statcast
- [ ] Import 2026 Statcast
- [ ] Compare schemas
- [ ] Generate import validation report

## Sprint 3 — Master Pitch Database
- [ ] Merge seasons
- [ ] Normalize columns
- [ ] Save master_pitch_database.parquet
- [ ] Save CSV preview
- [ ] Update manifest

## Sprint 4 — Master Game Database
- [ ] Create one row per game
- [ ] Separate pregame features from postgame results
- [ ] Validate scores and teams
- [ ] Save master_game_database.parquet

## Sprint 5 — Feature Engine
- [ ] Rolling team features
- [ ] Rolling pitcher features
- [ ] Bullpen fatigue features
- [ ] Team identity features

## Sprint 6 — Research Lab
- [ ] Hypothesis tracking
- [ ] Team predictability reports
- [ ] Bullpen identity reports

## Sprint 7 — Models
- [ ] Winner model
- [ ] Totals model
- [ ] Team-runs model
- [ ] Confidence engine
- [ ] Explainability engine

## Sprint 8 — Daily Predictions
- [ ] Daily slate import
- [ ] Prediction report
- [ ] CSV export
- [ ] Dashboard-ready output

## Sprint 9 — Critic Engine
- [ ] Grade predictions
- [ ] Store postgame reviews
- [ ] Feed lessons into knowledge library
