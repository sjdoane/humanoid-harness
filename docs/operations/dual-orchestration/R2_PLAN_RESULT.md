# R2 protocol proposal: collected, not run-approved

| status | current truth |
|---|---|
| progress | The read-only Sol planner finished at 17:20:01 UTC on 2026-09-05. |
| bottleneck | Its fixture protocol is unreviewed; Fable subsequently proposed a cycle-first shared integration. |
| next step | Review the minimum execution path against cycle 1 before dispatching implementation or host probes. |

- Baseline: `754e869c618dbe4f1b43c4f1a513c023ec139306`.
- Run: `.orchestration/sol-runs/20260905T171030Z-de04d66f-5dcd-4117-ad96-c64fb4f0e357`.
- Requested: `gpt-5.6-sol`, `max`, read-only planner, no delegation.
- Terminal: `SUCCEEDED`, exit 0; no writer lease required.
- Watcher: terminal observed; attached in the launch call with a deadline
  measured from the immutable launch time. No timeout occurred.
- Packet: `TASK-R2-runtime-plan.md`, SHA-256
  `7d7d59bad2a75e4195a70ef830012f38affcd6bd5f9393868e19746df118df9d`, 2,851 bytes.
- The full proposed protocol is retained in that run's `final.txt`.

## Proposed slices

1. Correct denial/control tests and bind canary authority to the tested runtime.
2. Run one exact numerical batch through the real isolated worker lifecycle.
3. Prove categorical failures, identity checks, deadlines, and cleanup.

Proposed host bounds: one child at a time, at most 33 launches, 300 seconds
overall, loopback-only permissive network control. These are **proposals**,
not resource approval. The proposed CPU/address-space controls need review
against the host's actual guarantees and the enclosing deadline.

Public candidate execution and dynamic calibration remain refused. The planner
did not run tests, probes, reward code, simulation, or training. No R2 finding
is closed by a plan. See `CYCLE_CONVERGENCE_PROPOSAL.md` for the new peer request.
