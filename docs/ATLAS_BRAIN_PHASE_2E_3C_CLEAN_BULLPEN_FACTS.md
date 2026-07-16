# ATLAS Brain Phase 2E.3C — Clean Bullpen Pregame Facts

## Purpose

Preserve objective bullpen facts that existed before first pitch while
excluding handcrafted baseball opinions and same-game outcomes.

## Canonical source

`/content/drive/MyDrive/Project_Atlas/data/pregame/bullpen/bullpen_pregame_state.parquet`

## Clean output

- Rows: 12,350
- Columns: 49
- Seasons: 2024, 2025, 2026
- Duplicate team-games: 0
- Future games used: no
- Same-date games used: no
- Predictions created: no

## Raw fields preserved

- `prior_bullpen_date`
- `days_since_prior_bullpen_date`
- `bullpen_pitches_prior_1_dates`
- `bullpen_games_used_prior_1_dates`
- `bullpen_pitches_prior_2_dates`
- `bullpen_games_used_prior_2_dates`
- `bullpen_pitches_prior_3_dates`
- `bullpen_games_used_prior_3_dates`
- `bullpen_pitches_prior_5_dates`
- `bullpen_games_used_prior_5_dates`
- `bullpen_pitches_prior_7_dates`
- `bullpen_games_used_prior_7_dates`
- `bullpen_whiffs_prior_3_dates`
- `bullpen_whiffs_prior_5_dates`
- `bullpen_strikeouts_prior_3_dates`
- `bullpen_strikeouts_prior_5_dates`
- `bullpen_walks_prior_3_dates`
- `bullpen_walks_prior_5_dates`
- `bullpen_hits_allowed_prior_3_dates`
- `bullpen_hits_allowed_prior_5_dates`
- `bullpen_whiff_per_pitch_prior_5_dates`
- `bullpen_whiff_per_pitch_season_prior_mean`
- `bullpen_strikeout_per_pitch_prior_5_dates`
- `bullpen_strikeout_per_pitch_season_prior_mean`
- `bullpen_walk_per_pitch_prior_5_dates`
- `bullpen_walk_per_pitch_season_prior_mean`
- `bullpen_hits_per_pitch_prior_5_dates`
- `bullpen_hits_per_pitch_season_prior_mean`
- `bullpen_runs_per_pitch_prior_5_dates`
- `bullpen_runs_per_pitch_season_prior_mean`
- `bullpen_pitches_season_prior_mean`
- `bullpen_pitches_season_prior_std`
- `bullpen_walks_season_prior_mean`
- `bullpen_hits_allowed_season_prior_mean`
- `bullpen_consecutive_prior_usage_dates`
- `bullpen_recent_workload_zscore`
- `bullpen_snapshot_available`
- `specific_reliever_availability_known`

## Handcrafted fields blocked

- `bullpen_rest_recovery_score`
- `bullpen_workload_pressure_score`
- `bullpen_fatigue_score`
- `bullpen_state_label`

## Rule

The adapter measures bullpen state. It does not determine whether that state
is good, bad, tired, rested, strong or weak. Those relationships must be
learned from historical pregame facts and factual outcomes.
