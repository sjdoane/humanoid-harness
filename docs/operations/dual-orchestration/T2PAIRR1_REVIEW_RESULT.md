# Pairing repair closure

| status | current truth |
|---|---|
| progress | PAIR-01 canonical-byte equality and PAIR-03 source bindings close by independent inspection. |
| bottleneck | PAIR-02 observes helper outputs, not all downstream uses; its single mutation cannot establish each consumer regression. |
| next step | Return one narrow sink-level test repair to Fable. No competing source patch or study execution. |

- Verdict: **`REPAIRS_T2PAIRR1`** on
  `df3c41d5980ac31efb5024798e20e473f36fa17b`.
- Run: `.orchestration/sol-runs/20260906T124421Z-f2726334-bd50-4caa-a9f1-c512af01f0be`.
- Requested Sol/max, read-only; terminal `SUCCEEDED`, exit 0, 12:49:59 UTC.
- Final SHA-256: `ee7aec165c938c9f7c808a4dec1c987dd68cf1e8ce09e192f07785a107a8bafe`.
- Watcher: `terminal_observed`; timeout termination was not exercised.
- No reviewer tests, imports, fake runtime or simulation ran. The parent also
  inspected the actual capture/diff in the preceding checkpoint, not execution.

| Finding | Disposition | Exact source at reviewed commit |
|---|---|---|
| PAIR-01 | Closed: both projections validated, canonical bytes compared, verified bytes hashed; constructor alias negatives | `phase_b/contracts.py:1051`; `tests/phase_b/test_contracts.py:123` |
| PAIR-02 | Open: real PPO/scheduler paths run in the test, but action/minibatch spies capture helper returns, not actual consumer inputs | `phase_b/training.py:929,1118`; `tests/phase_b/test_pairing.py:103` |
| PAIR-03 | Closed: five protocol digests and the 9,770-byte pairing receipt bind exact committed source bytes | `reward_study/pairing.py:483`; `tests/reward_study/test_t2_artifacts.py:55` |

## Smallest remaining closure

- Keep helper calls intact while independently misrouting each downstream use.
- Observe epsilon actually supplied to `actor.act`, actual ordered PPO update
  inputs, and independently expected composition/reset/RSI sequences per slot.
- Require each consumer mutation to fail separately. Cross-arm equality alone
  can miss both arms making the same mistake; disabling the shared switch can
  fail first on empty action capture without testing other consumers.
- One parameterized regression is sufficient; no new experiment or generalized
  testing framework is requested. Fable retains its current source ownership.

Static inspection preserves the legacy unpaired path and distinct full-arm
identities. This is a coverage gap, not an observed corrupt training cohort.
Common draws do not imply identical trajectories or learned behavior.

The peer summary remains 1 failed / 1922 passed / 16 skipped. Its later historical
reward-receipt regeneration is not a reproduced full-suite pass. This verdict
grants no F3 dispatch, integration, runtime, resource or cohort authorization.
