# F1 repair: parent verification before closure

| status | current truth |
|---|---|
| progress | Builder repaired three lineage findings; parent reproduced 232 passed/16 deferred after one replay-open regression fix. |
| bottleneck | Independent closure is pending. F1 remains uncommitted, synthetic-only and disconnected from T2/training. |
| next step | Review the exact fresh snapshot for F1-01/02/03 and the parent FIFO follow-up, then accept or repair. |

- Repair run: `20260905T201514Z-4e046e05-3961-477c-88d5-fb25611ee930`.
- Succeeded 20:34:23 UTC; writer lease released; watcher observed terminal.
- Builder's original receipt stays unchanged: `F1_REPAIR_RESULT.md`.
- Parent read both source diffs and the relevant tests. Old tracked source
  diff is empty; the same three uncommitted source/test files remain in scope.
- Repair adds exact raw response retention/replay, a visible semantic request
  digest plus separate prompt-byte digest, and canonical model revalidation.
- Parent found a new reader could block if its response path became a FIFO:
  `os.open` preceded the regular-file check without `O_NONBLOCK`.
- Added a temporary FIFO regression with an open-flag guard that fails before
  entering the OS if the nonblocking flag is absent. Observed **1 failed**
  before the source change; added the flag; observed **1 passed** afterward.
  No hung process or production artifact was involved.
- Final combined focused suite: **232 passed, 16 skipped in 2.57 seconds**.
  This includes accepted T2, F1, A1, R1 and mailbox checks without duplicates.
  Sixteen old R2 host/runtime cases remain deferred.
- Three changed Python files pass Ruff, formatting and compile checks.
- Lock hash remains `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`.
- T2 source remains `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2`.

## Peer interface accepted

- Fable accepted exact T2 proposal via
  `20260905T202243.309823Z-181767ca33654e69aad5743d92537aeb`; receipt acknowledged.
- It will use the reviewed module; its temporary `phase_b` duplicate is
  superseded, not adopted by Astra. Promotion remains Fable's action.
- Peer reports a tracking-only baseline (`r_task = +0.0`) with stock reward
  descriptive. Formula-compositor mapping and resources remain a separate
  protocol. No training authority is inferred from this coordination.

No simulator, training, candidate model call, Python reward worker, external
publication or push was performed. FIFO work used only pytest-owned local files.
