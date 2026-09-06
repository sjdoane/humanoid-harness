# Composition cycle 1 record

| status | current truth |
|---|---|
| progress | The prompt-only candidate was copied byte-for-byte, passed the contract validator, ran on all `20` frozen seeds for `1,000` steps, and matched all `20/20` determinism replay hashes. |
| bottleneck | The candidate fell in `20/20`, never executed `simple`, and spent `100%` of every episode's slow third in `medium`; it did not improve either switching baseline. |
| next step | Cycle 2 is prepared from the frozen cycle-1 report with the launch-note steering sentence recorded verbatim. Fable reviews and commits before any later designer or write slice. |

## Candidate provenance

| item | value |
|---|---|
| designer | `sol-designer-20260905-e003d1`; launcher requested read-only; observed reads are audit-log-only, not OS-enforced isolation |
| designer run directory | `.orchestration/sol-runs/20260905T183122Z-b0788053-4fd7-4887-af1d-8c2ef411c93b` |
| source | `.orchestration/oracles/cycle_1_candidate.json` |
| frozen copy | `cycles/cycle_1/oracle_1.json` |
| raw file SHA-256 | `32bfc555ffc578d4cf8f75823acc1ba98db6621b2bcd0204f0725b0bf70af029` for both paths; `cmp` matched |
| canonical oracle SHA-256 | `9d0929cb785429776283e15765bdb3c3f19c60f1a0a3166317909f5f7bef381a` |
| contract validation | pass; oracle ID `cycle_1_candidate`; no oracle edit |
| precondition | clean `2fb31dcc85a23d1d1ed0b8ab3c86b30d460d7d51` |

## Evaluation result

| metric | cycle-1 candidate |
|---|---:|
| episodes / horizon | `20 / 1,000` steps |
| median mean absolute speed error | `3.2215072393 m/s` |
| falls | `20/20` |
| first fall boundary | median `332.5`; range `322`-`357` |
| median controller switches | `3` (`14` episodes had `3`; `6` had `1`) |
| median behavior time | expert `299`; medium `701`; simple `0` steps |
| slow-third fraction `[300,600)` | expert `0`; medium `1`; simple `0` in every episode |
| median descriptive stock task return | `2,995.0573738434` |

- Every episode switched from `expert` to `medium` at step `300`; its first fall followed `22`-`57` steps later.
- Fourteen episodes also entered and exited `medium` recovery at steps `11`-`14`. These early switches did not precede the first fall by at most `100` steps.
- The candidate matched `playback` and `handwritten` at `20` falls, while median MAE worsened by `0.0411646540 m/s` and `0.0152911780 m/s`, respectively.
- Relative to `single_fast`, both fall count and MAE worsened. Relative to `single_slow`, fall count worsened while MAE improved by `0.0856085912 m/s`. No combined ranking was predeclared.

## Receipts

| artifact | SHA-256 / result |
|---|---|
| runtime fingerprint | `186fd2f4aa6b9ed9a0eb3de73faa0d2bbcb392a4fe52f992b6b443fbf33d2621`; unchanged from cycle 0 |
| `report_1.json` | `480d5b3521ce66593e0258f3f42904f969c72b83141b330bc7ff4759144c25f0` |
| `report_1.md` | corrected render `116e2a836ea5f2a0ae6b596e7005c0e0dbee701ad7a75d468c3d73d92817207b`; the pre-review bytes read by the cycle-2 designer were `88cefeeeb5cfe47f40d4623233b3a76a73e18f7ea0b70a7b186369b04029170a` and remain bound in `designer_provenance.json` |
| `scientific_receipt_v2.json` | `5a5a996c28423dd6c33de3b33c527de43825aeedccb485338d94ad033882cecc`; deterministic scientific chain |
| `telemetry_v1.json` | `ddec44f6de097c2514a3d29a94474d20e03b7bfa5807d7f98fa0b2aeb5bd4d51`; operational timestamps, host fields, wall times, and receipt bindings |
| `designer_provenance.json` | `5abfca31c5e87d1d84526edc8b8c66a943f886b0333e4dcd26ffc3a1351e58a4`; sanitized audit-log-only receipt |
| ignored trace index | `4d2c419140b71d1cc13f04c6af802d9fb569aaf690621cc17cd99ab5f4057097`; `20` entries; cycle directory `14M` |
| determinism replay | `20/20` hashes matched; `7.4264 s` inside the report |
| evaluation wall time | `15.0420 s` inside the report; `15.53 s` process wall including startup and writes |

## Validation

| check | result |
|---|---|
| independent artifact audit | pass: oracle bytes and contract, canonical report, exact `20` seeds, all `20` indexed trace bytes and hashes, recomputed switch/fall/slow-third diagnostics, and cycle-2 bindings |
| focused harness tests | `13 passed` in `0.59 s` process wall |
| Ruff lint / format | pass; `250` files already formatted |
| `git diff --check` | pass |
| builder wall time | `14m18s` from launcher start through the final lease, scope, and diff check |

## Cycle-2 preparation

| item | value |
|---|---|
| steering source | Fable launch note; no steering was supplied by a human |
| steering text, verbatim | `No additional steering text was supplied.` |
| `cycles/cycle_2/designer_prompt.md` | `99b1714912970a98d9436dfbbf9dee2f975e32d574d8a36d81ff085d02c77dab` |
| `cycles/cycle_2/expected_inputs.json` | `4048d6477c7864c838c5d87717c016827d5e62dc8c8569a0a2977e77702c6d6b` |
| prior report binding | `480d5b3521ce66593e0258f3f42904f969c72b83141b330bc7ff4759144c25f0` |

## Claim ceiling

This is one exploratory controller-switching cycle on the frozen plain `Humanoid-v5` runtime. It supports no oracle-quality, generalization, frozen-tracker, reference-following, task-reward, naturalness, robustness, or humanoid-competence claim.
