# Immutable project history

`MIGRATION_MANIFEST.json` preserves the exact old-repository evidence migration
and the initial new-review scaffold.

The review tables are live research records, so their original empty bytes are
stored under `initial_review_scaffold/`. Manifest schema v2 verifies those
archived bytes while allowing the live tables to advance. This does not alter
the 104 copied legacy files or their recorded hashes.
