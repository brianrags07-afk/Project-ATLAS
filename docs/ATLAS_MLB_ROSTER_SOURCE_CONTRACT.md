# MLB roster source contract

ATLAS acquires three official MLB Stats API fact sets: the season team directory,
dated active/40-man roster snapshots, and team transaction records. Every normalized
row retains retrieval time, source precision, and a hash of the raw source record.
Transactions retain generic, sending, and receiving team fields separately so
departures and arrivals remain distinguishable without parsing narrative text.
MLB may reuse one transaction ID across several source rows. ATLAS preserves all
of them and assigns a stable ID/hash/occurrence `transaction_key`; semantic event
conversion must reason over those rows and must not assume transaction ID alone
is a row-level primary key.
Roster endpoints may likewise repeat an entry or omit a usable person ID. ATLAS
retains each source row with a hash-plus-occurrence `roster_key` and exposes
`player_identity_known`; unidentified rows remain auditable but cannot become
player events until a separately sourced identity resolution succeeds.

This layer is deliberately lossless and does **not** translate transaction prose into
membership or availability state. Historical transaction fields are often day-precise,
not first-pitch-precise. They therefore carry `pregame_time_known = false`; a later,
tested conversion layer must defer ambiguous same-day moves until a subsequent game.

Raw payloads and normalized outputs belong in immutable build/staging paths. They do
not overwrite certified schedules, master datasets, or canonical roster artifacts.
