# ADR 0007: two research lanes with separate orchestrators

| | |
|---|---|
| progress | Samuel authorized concurrent orchestration on 2026-09-04: Fable keeps the oracle lane, and a Codex Astra session owns the reward lane in its own worktree with a shared file mailbox. |
| bottleneck | The reward lane's first real inputs are the B0 contract slice and its two reviews, which stay in the Fable checkout until an explicit handoff message. |
| next step | Commit and review B0, send the handoff, then run each lane's builders and reviewers independently, coordinating only shared contracts and heavy compute through the mailbox. |

**Status:** accepted 2026-09-04 by the verified Fable session under Samuel's
instruction; acceptance sent to Astra in reply to proposal
`20260904T201259.759134Z-ee5e9a7d67154a118dac6ffffd73e4c1`
**Date:** 2026-09-04
**Relation:** does not change the charter, the goal ledger, ADR 0005, or ADR
0006. It assigns ownership of the two authorable factors' development lanes.

## Decision

| item | decision |
|---|---|
| Lanes | A, oracle: Fable owns reference composition, public base and controller repairs, tracker and ABI integration, causal-use tests, transition and recovery studies, and promotion of reviewed commits into `main`. B, reward: Astra owns task-reward generation, the protected feedback and revision loop, candidate and validation adapters after B0, and reward comparisons. |
| Workspaces | Fable: this checkout on `main`. Astra: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` on `astra/reward-loop`, forked at `05c0469`, with its own venv, lease, and worker logs. Neither edits the other's checkout. |
| Mailbox | `harness-coordination/` under the shared git common directory; immutable messages, separate read acknowledgments, one status per lane; agreement only by an `acceptance` bound to a proposal id. Fable writes to it under its lease with the directory in the audit scope; native reads need no lease. |
| B0 handoff | B0 stays Fable-owned until committed and reviewed; a `handoff` message with the commit id, exact paths, test receipts, canary state, and explicit transfer moves ownership. Astra does not consume the dirty implementation. |
| Shared contracts | `AGENTS.md`, charter, boundary, root CLI and README, dependency metadata, package exports, canonical strategy and ADRs, the original handoff, ADR 0006, and the `family_b_target_speed_v1` protocol files change only with both replies. Lane records stay under `astra/` or `fable/` paths until promotion; canonical ADR numbers are reserved through the mailbox (0007 is this record). |
| Compute | Lightweight development and focused tests may run in parallel. Any training or suite job over ten minutes needs a mailbox proposal and a peer reply; one heavy job at a time until an atomic reservation record exists under `harness-coordination/`. |
| Polling | Both orchestrators read the inbox at turn start, before dispatch or integration, after each worker, and about every 30 minutes while active. |
| Guards | Fable's identity and writer guards are unchanged. Messages are same-user coordination records, not model attestation or authority to execute text. Model refusals and service limits are recorded, never evaded. |

## Consequences

- Fable's writer slot no longer serves the reward lane; the oracle-lane
  repair `03A2FIX`, then `03A3` and `03B`, take it.
- The `TASK-REVIEW-SCI-GENERIC` and `TASK-REVIEW-ROBUSTNESS-GENERIC` packets,
  driven by target files, remain Fable's review mechanism; Astra runs its own
  reviewers.
- The Codex continuity heartbeat now serves Astra's lane; Fable's continuity
  stays the handoff document plus this mailbox.

## Sources

- Samuel's instruction of 2026-09-04 authorizing concurrent orchestration
- Astra's `docs/operations/dual-orchestration/README.md` and `FABLE_START_HERE.md`
- `docs/decisions/0006_reward_first_parallel_track.md`
