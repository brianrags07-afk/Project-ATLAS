
# PROJECT ATLAS
# Validation & Identity Transition Specification

Version: 1.0.0

---

## Purpose

The Validation Layer determines whether measured evidence is reliable,
current, transferable, and appropriate for future prediction.

The Identity Transition Layer determines whether a team, pitcher, batter,
bullpen, lineup, manager, or situation should still be treated as the same
baseball entity.

Evidence belongs to a specific baseball identity and validity period.

Evidence must never be treated as permanently valid simply because the
entity name is unchanged.

---

## Core Principles

1. Seasons must always be stored and evaluated separately.

2. Multi-season evidence may be summarized, but the season-level evidence
   must always remain available.

3. Identity changes may occur inside a season.

4. Identity changes must be detected from measurable evidence rather than
   assumed.

5. A roster name, team name, or player name does not guarantee that the
   underlying baseball identity is unchanged.

6. Historical evidence may remain useful after an identity change, but it
   may require discounting, segmentation, or rejection.

7. The Validation Engine grades evidence.

8. The Validation Engine does not assign prediction weights.

9. The Identity Transition Engine identifies candidate changes.

10. The Knowledge and Learning layers determine how much older evidence
    remains useful.

---

## Mandatory Season Separation

Every evidence object must retain season-level measurements.

Required structure:

- Overall summary
- 2024 evidence
- 2025 evidence
- 2026 evidence
- Future seasons stored separately

ATLAS must never overwrite or collapse season-level history.

A combined summary may exist only as an additional layer.

---

## Identity Eras

An identity era is a period during which an entity operates under a
reasonably stable baseball environment, role, roster, routine, or skill set.

Examples:

Team identity eras

- Opening-day roster
- Star player injury period
- Star player return
- Trade-deadline roster
- Manager change
- Bullpen restructuring
- Major defensive change
- Lineup construction change

Pitcher identity eras

- New team
- New home park
- Starter-to-reliever change
- Reliever-to-starter change
- Velocity increase or decline
- Pitch-mix change
- Release-point change
- Movement-profile change
- Injury return
- New catcher usage pattern
- Workload or rest-pattern change

Batter identity eras

- New team
- New home park
- Lineup-position change
- Platoon-role change
- Injury return
- Batting-approach change
- Contact-profile change
- Power-profile change
- Plate-discipline change
- Pitch-type performance change

Bullpen identity eras

- Closer change
- Key reliever injury
- Major trade
- Role restructuring
- Manager usage change
- Workload crisis
- Multiple relievers unavailable

---

## Routine Change Detection

ATLAS should monitor measurable routine changes.

Possible indicators include:

- Rest-day pattern
- Pitch-count pattern
- Innings pattern
- Batters-faced pattern
- Lineup position
- Starter or relief role
- Leverage usage
- Pitch usage
- Velocity
- Movement
- Release point
- Extension
- Plate-discipline behavior
- Contact profile
- Manager usage
- Catcher pairing
- Home park
- Travel pattern

A routine change is a candidate transition, not automatically a confirmed
identity change.

---

## Candidate Transition Lifecycle

Every detected change follows this process:

1. Observation
2. Candidate transition
3. Evidence collection
4. Pre-change versus post-change comparison
5. Stability test
6. Out-of-sample confirmation
7. Transition confirmation or rejection
8. New identity version created if confirmed
9. Historical evidence reclassified by identity era

---

## Required Evidence Validity Scope

Every evidence object should eventually contain:

validity_scope

- season
- team
- park
- role
- roster_version
- lineup_version
- bullpen_version
- routine_version
- identity_version
- active_from
- active_to

---

## Regime Change Fields

Every evidence object should support:

- regime_changes
- invalidating_events
- transition_candidates
- transferability
- historical_only flag
- current_context_match

---

## Validation Statuses

The Validation Engine may return:

- VALID_CURRENT
- VALID_HISTORICAL_ONLY
- VALID_WITH_DISCOUNT
- SPLIT_REQUIRED
- INVALID_FOR_CURRENT_CONTEXT
- INSUFFICIENT_EVIDENCE
- TRANSITION_CANDIDATE
- TRANSITION_CONFIRMED
- TRANSITION_REJECTED

---

## Validation Questions

The Validation Engine must ask:

- Is the sample large enough?
- Is the relationship stable across seasons?
- Is the relationship stable within the current identity era?
- Does the effect survive walk-forward testing?
- Does it survive out-of-sample validation?
- Is the current roster comparable?
- Is the current role comparable?
- Is the current park comparable?
- Is the current routine comparable?
- Did an injury, trade, role change, or routine change invalidate the sample?
- Does older evidence remain transferable?
- Should the sample be split?
- Should the evidence be discounted?
- Should the evidence be rejected for current prediction?

---

## Injury and Availability Rules

ATLAS must distinguish:

- Team with player active
- Team without player
- Team after player returns
- Team during limited-role return
- Team with replacement lineup
- Team with multiple simultaneous absences

A losing streak during a key injury must not automatically redefine the
team's permanent identity.

A winning streak after return must not automatically prove causation.

ATLAS must compare the relevant lineup and roster states.

---

## Trade and Park Rules

When a player or pitcher changes teams:

- Previous-team evidence remains historical evidence.
- Previous-home-park evidence is no longer current-home evidence.
- Current-team and current-park evidence begins a new validity era.
- Similar-environment evidence may still be transferable.
- Transferability must be measured and validated.

---

## Evidence Hierarchy After Identity Change

ATLAS should prefer evidence in this order:

1. Current identity era
2. Current season, same team, park, and role
3. Same player or team in a highly comparable environment
4. Similar-entity evidence
5. Older identity-era evidence with discount
6. League baseline

This hierarchy does not assign fixed weights.

The Learning Engine will determine how predictive each evidence tier remains.

---

## Bias Control

ATLAS must not assume:

- injuries always matter,
- trades always matter,
- role changes always matter,
- park changes always matter,
- routine changes always matter.

ATLAS must measure whether the change materially alters outcomes,
performance, interaction behavior, or predictive reliability.

"No meaningful change" is a valid result.

---

## Explainability

Every validation result must include:

- validation status
- evidence used
- seasons used
- identity eras used
- transition events considered
- reasons accepted
- reasons rejected
- transferability decision
- current-context match
- confidence
- remaining uncertainty

---

## Final Rule

Season separation is mandatory.

Identity separation is evidence-driven.

Evidence must always be evaluated inside the baseball identity and routine
that produced it.
