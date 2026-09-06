# T2PAIRR2 sink-level pairing regression

| status | current truth |
|---|---|
| progress | PAIR-02 now has exact sink-level expectations and 10 independently exercised mutation cases across action, minibatch, composition, reset, and RSI routing. |
| bottleneck | This builder result is not independent closure. T2 execution and F3 dispatch remain withheld. |
| next step | Fable reviews and commits this test-only repair, then sends the exact commit to Astra for narrow PAIR-02 re-review. |

## Preflight and scope

| item | observed value |
|---|---|
| launch base | clean `f9f032e` |
| writer lease | `CLAIMED` by `sol-builder-20260906-t2pairr2`; requested `gpt-5.6-sol`, role `builder`, exact declared scope |
| source review | Astra verdict `REPAIRS_T2PAIRR1`; PAIR-01 and PAIR-03 closed, PAIR-02 open only at the sinks |
| production routing | inspected and unchanged; no defect found |
| changed implementation path | `tests/phase_b/test_pairing.py` only |
| claim ceiling | regression coverage only; no contaminated cohort is claimed or refuted |

The existing two-arm receipt test remains unchanged. The repair replaces its
helper-return production spy with observers at the values actually consumed by
`actor.act`, every NumPy-to-Torch PPO update input, and each fake environment's
assignment/reset boundary. Expected values are regenerated independently for
each exact arm plan and each environment slot; cross-arm equality is retained
but is no longer sufficient.

## Independent mutation matrix

| route | exact observed sink | single sink mutation applied to both arms | disabled-router check |
|---|---|---|---|
| action | Every epsilon passed to `FullAuthorityActor.act` | Roll the four environment rows after the paired helper returns | Fails the action assertion with nonempty actor captures |
| minibatch | Every ordered observation, pre-tanh action, old log-probability, advantage, and return array consumed by real `_ppo_update` | Roll every consumed array at the Torch boundary after the paired ordering helper returns | Fails the minibatch assertion with nonempty PPO captures |
| composition | Per-slot block sequence returned to environments 0 and 1 | Rotate each returned block after the correct assignment call | Fails against each slot's independently generated schedule |
| reset | All four environments' exact reset ledgers plus routed slot sequence | Deliver the opposite composition slot's block after the correct per-slot call, identically in both arms | Fails exact per-slot reset expectations |
| RSI | Exact assignment sequence delivered to environments 2 and 3 | Deliver the opposite rehearsal slot's assignment after the correct per-slot call | Fails exact per-slot RSI expectations; the production ledger validator also rejects it |

The parameterized regression runs each of the five sink mutations separately,
then disables `_paired_routing_enabled` and asserts each route separately. Every
case requires complete action, PPO, composition, reset, and RSI captures before
the expected route-specific assertion may fail. In the five sink-mutation cases,
the production action and minibatch helpers are still called twice per arm and
their outputs are returned unchanged to the downstream mutation boundary.

## Verification

| check | observed result |
|---|---|
| Pairing file | final readback `15 passed in 1.03s`; parameterized mutation matrix separately `10 passed in 5.20s` |
| Phase B plus reward-study tests | final-byte run `254 passed in 245.33s` |
| Full suite with the named forbidden receipt test deselected | final-byte run `1886 passed, 18 skipped, 1 deselected, 44 failed in 428.16s` |
| Sandbox baseline replay | all 44 recorded node IDs failed in `1.17s`; with full-suite failure cardinality 44, the sets are equal |
| Repository-wide Ruff lint / format | passed / all `409` files formatted in the final four-path tree |
| `git diff --check` | passed after the final four-path documentation audit |
| Builder wall time | `50m` from lease acquisition through final validation and handoff readback |

The excluded test was exactly
`tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
Its receipt was not executed or regenerated and remains 4,150 bytes at
`7ef1f9780092108872b3abe7cacabb254801718596aa2de1067db8074ead7efc`.
No candidate call, simulator smoke, real training, cohort, protected evaluation,
or behavioral evaluation ran. No execution/study/F3 seal or pairing receipt was
changed because production routing did not change.

Fable resume: inspect the five actual sink observers, independently generated
per-arm/per-slot expectations, all 10 parameter cases, complete-capture guards,
and the unchanged production diff. Commit this result with the test, README, and
current handoff updates, then request Astra's narrow PAIR-02 re-review. Keep F3
dispatch and all T2 execution withheld.
