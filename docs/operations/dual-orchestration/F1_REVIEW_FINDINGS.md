# F1 review: three bounded repairs

| status | current truth |
|---|---|
| progress | Formula math and input gates passed review; the separate T2 input module is accepted. |
| bottleneck | REJECT_F1: response replay, prompt identity and forged schema labels remain incorrect. |
| next step | Repair the two formula-lineage modules and their test file; independently recheck these findings. |

- Run: `.orchestration/sol-runs/20260905T193917Z-382d4a3e-5e85-4229-9d82-a7958035231b`.
- Finished 19:48:21 UTC, requested Sol/max read-only. Watcher observed terminal.
- Full review: that run's `final.txt`; nine source/doc hashes and lock matched.
- Parent independently reverified all nine hashes before any repair.
- Exact old repair files retained under
  `.orchestration/f1-pre-repair-20260905/`, using their original relative paths.

| ID | Reproduced by reviewer | Required repair |
|---|---|---|
| F1-01 | Malformed response publishes a rejection receipt but loses the exact response bytes. | Retain bounded raw responses, bind their identity/count and reverify on replay. |
| F1-02 | The required response packet hash is absent from the rendered prompt; tests supply it out of band. | Explicit non-self-referential request identity plus separately verified prompt bytes, or a retained request envelope. |
| F1-03 | `model_copy` with forged packet or bindings `kind` passes public entry points. | Fully revalidate canonical copies at public boundaries; use those copies, not the original objects. |

The reviewer ran 27 core tests and 17 T2 tests successfully. Formula-loop tests
reported 20 passed and 12 setup errors because that read-only sandbox lacked a
writable pytest temporary directory. Source findings and in-memory probes are
retained; those 12 tests are not reported as passed by the reviewer. Parent's
earlier writable focused reproduction remains 203 passed/16 old R2 deferrals.

F1 source remains uncommitted/unaccepted. Existing Python execution refusal,
absent compositor and synthetic/unverified response-origin labels stay intact.
