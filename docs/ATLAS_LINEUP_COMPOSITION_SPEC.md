
# PROJECT ATLAS
# Lineup Composition & Pregame Grading Specification

Version: 1.0.0

---

## Purpose

The Lineup Composition Layer represents the exact pregame lineup as a
combination of individual player identities, batting order, matchup fit,
availability, and environment.

ATLAS must preserve both:

1. Individual player grades
2. Combined lineup grades

The combined lineup must never erase the individual contributions that
created it.

---

## Core Rule

A team does not have one permanent offensive identity.

Its pregame offensive identity depends on:

- Which players are active
- Which players are absent
- Exact batting order
- Platoon configuration
- Opposing starter identity
- Opposing pitch mix
- Opposing bullpen composition
- Park
- Weather
- Umpire
- Rest
- Travel
- Current player identity eras

---

## Pregame Safety

Only information known before first pitch may be used.

Allowed:

- Confirmed lineup
- Published batting order
- Active roster
- Known injuries
- Announced starter
- Available bullpen
- Pregame weather
- Park
- Umpire when announced
- Rest and travel known before the game

Not allowed:

- Final game results
- In-game substitutions
- Actual bullpen usage
- Postgame injuries
- Outcome-derived lineup strength
- Any information learned after first pitch

---

## Individual Batter Grade

Every batter should retain separate pregame grades for:

- Overall offense
- Contact
- Power
- Plate discipline
- Strikeout resistance
- Walk creation
- Ground-ball profile
- Fly-ball profile
- Line-drive profile
- Fastball performance
- Four-seam performance
- Sinker performance
- Cutter performance
- Slider performance
- Sweeper performance
- Curveball performance
- Changeup performance
- Splitter performance
- High velocity
- Low velocity
- Pitch movement
- Pitcher handedness
- Park fit
- Weather fit
- Lineup position
- Expected bullpen matchup
- Current identity era
- Current health and availability
- Confidence and sample size

No individual grade is assumed predictive.

Every grade must be validated for the player and target.

---

## Combined Lineup Scorecard

The combined lineup object should include:

- overall_offense_grade
- contact_grade
- power_grade
- discipline_grade
- strikeout_resistance_grade
- walk_pressure_grade
- fastball_grade
- slider_grade
- breaking_ball_grade
- offspeed_grade
- velocity_grade
- movement_grade
- platoon_grade
- park_fit_grade
- weather_fit_grade
- starter_matchup_grade
- bullpen_matchup_grade
- lineup_depth_grade
- lineup_order_grade
- top_order_grade
- middle_order_grade
- bottom_order_grade
- bench_support_grade
- volatility_grade
- confidence

The combined grade is a summary, not a replacement for component grades.

---

## Batting Order

ATLAS must preserve the exact batting order.

The same nine players in a different order are different lineup objects.

Batting order affects:

- Expected plate appearances
- First-inning scoring
- Run creation
- RBI opportunity
- Total-base opportunity
- Stolen-base opportunity
- Bullpen exposure
- Pinch-hit probability
- Player props
- YRFI and NRFI
- First Five outcomes

Every lineup object should carry a unique lineup version.

---

## Lineup Versioning

A new lineup version should be created when any of the following change:

- Player added
- Player removed
- Batting order changed
- Platoon configuration changed
- Injury status changed
- Limited role changed
- Returning player activated
- Trade acquisition added
- Lineup position materially changed

Lineup evidence must be connected to the lineup version that produced it.

---

## Combined Pregame Matchup Object

The lineup scorecard should be compared with:

- Opposing starter identity
- Starter pitch mix
- Starter velocity
- Starter movement
- Starter command
- Starter handedness
- Starter times-through-order profile
- Expected bullpen composition
- Bullpen availability
- Bullpen fatigue
- Park
- Weather
- Umpire
- Rest
- Travel
- Series context

The resulting object is a Pregame Matchup Composition Object.

---

## Target-Specific Grades

Combined grades must remain target specific.

Examples:

Moneyline lineup grade

Totals lineup grade

YRFI lineup grade

First Five lineup grade

Batter prop grade

Pitcher strikeout-opposition grade

A lineup may grade strongly for one target and weakly for another.

---

## Team-Specific Learning

Lineup validation must be local before global.

ATLAS must learn separately for each team:

- Which lineup traits matter
- Which batting orders matter
- Which players materially change identity
- Which starter matchups matter
- Which bullpen matchups matter
- Which environmental effects matter

No league-wide lineup weight may be imposed automatically.

---

## Interaction Examples

ATLAS may investigate:

High Contact Lineup
x
Power Pitcher

Patient Lineup
x
Poor Command Pitcher

Fly-Ball Lineup
x
Ground-Ball Pitcher

Fastball-Crushing Lineup
x
Four-Seam Heavy Pitcher

Slider-Weak Lineup
x
Slider Secondary Pitcher

Weak Bottom Order
x
Deep Bullpen

Strong Top Order
x
First-Inning Vulnerable Pitcher

Every interaction is measured and validated.

No interaction is assumed important.

---

## Combined Score Construction

ATLAS must not permanently hard-code weights such as:

- lineup = 40 percent
- starter = 35 percent
- bullpen = 25 percent

Weights must be learned separately by:

- Team
- Lineup version
- Target
- Pitcher identity
- Bullpen identity
- Environment
- Season
- Identity era

A combined score must always remain decomposable into its individual inputs.

---

## Explainability

ATLAS should be able to explain:

- Which players increased the lineup grade
- Which players decreased the lineup grade
- Which batting-order positions mattered
- Which pitch types created advantages or weaknesses
- Which starter interactions mattered
- Which bullpen interactions mattered
- Which environmental interactions mattered
- How confident each component was
- Which historical games supported the conclusion

---

## Required Lineup Object Structure

Every pregame lineup object should eventually contain:

metadata

- game_pk
- game_date
- team
- opponent
- lineup_version
- confirmed_at
- source
- pregame_safe

players

- player_id
- player_name
- batting_order
- position
- handedness
- availability
- identity_version
- individual_grades
- confidence

combined_grades

matchup_grades

starter_interaction

bullpen_interaction

environment_interaction

validation

traceability

---

## Final Rule

ATLAS must preserve every individual score while also evaluating the
combined lineup as a single pregame baseball identity.

The lineup grade summarizes the players together.

It never replaces them.
