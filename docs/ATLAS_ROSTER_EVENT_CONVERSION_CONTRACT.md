# Roster source-to-event conversion contract

Opening roster snapshots establish organization membership only for known player
IDs. Active and 40-man rows collapse to one semantic event while retaining all
source hashes and row counts. Their historical availability begins at midnight
UTC after the prior-day snapshot.

Transactions create membership events only from explicit structured `fromTeam`
and `toTeam` direction **and** an approved organization-changing code: `TR`
(trade) or `CLW` (claimed off waivers). Direction on assignments and status
changes may reference affiliates and does not prove an organization change.
Repeated API rows collapse to one semantic event with
complete lineage. Because historical transaction dates are day-precise, events
become eligible at midnight UTC on the following day.

Narrative descriptions never determine injury, option, release, suspension, or
availability state. Missing identities, missing dates, and nondirectional records
are quarantined for separately sourced and tested semantic resolution.
