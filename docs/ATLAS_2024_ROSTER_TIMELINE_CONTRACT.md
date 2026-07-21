# ATLAS 2024 Roster and Player-Team Timeline Contract

## Purpose

ATLAS reconstructs the roster state known before every 2024 regular-season
game. It does not infer a roster, trade, injury, or availability state from a
player's later box-score or Statcast appearance.

## Canonical event grain

One row represents one sourced roster-state change for one player and team.
Required identity and lineage fields are `event_id`, `effective_at`, `season`,
`team`, `player_id`, `event_type`, `source`, and `source_retrieved_at`.

State changes may update `organization_member`, `active_roster`, `available`,
`injury_status`, and `roster_status`. Null state values mean “unchanged,” not
false. A trade must be represented by a removal event for the former team and
an addition event for the new team; the history is never rewritten in place.

## Pregame rule

A game snapshot may consume an event only when both its effective time and the
time ATLAS obtained the source are no later than scheduled first pitch. The
output preserves the last event and source timestamps so this rule can be
audited. Unknown history remains unknown and blocks certification.

## Derived artifacts

- roster event ledger: immutable sourced changes
- team pregame roster snapshot: game/team/player grain
- player-team timeline: intervals derived from the event ledger
- availability timeline: active, injured, optioned, suspended, or otherwise
  unavailable states

These artifacts feed lineup versions, bullpen availability, team identity,
player identity, trade/home-park effects, and pregame game cards. They never
replace the source event ledger.

## 2024 certification gate

Before 2024 roster intelligence can be frozen, ATLAS must prove:

- unique event IDs and valid UTC timestamps;
- source and effective chronology for every event;
- an opening state or explicitly documented unknown interval for every team;
- no post-first-pitch information in a pregame snapshot;
- explicit handling of trades, IL moves, activations, options, call-ups,
  releases, suspensions, and roster removals;
- reproducible counts and hashes for the event ledger and snapshots;
- regular-season game coverage against the certified 2024 schedule.

The initial Python foundation is source-agnostic. An MLB source adapter and
checksum-bound 2024 fixture must be added before any generated roster artifact
is promoted from staging.
