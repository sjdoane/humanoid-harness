# Composition cycle 2 record

| status | current truth |
|---|---|
| progress | The byte-exact cycle-2 candidate passed contract validation, ran for `1,000` steps on all `20` frozen seeds, and matched all `20/20` determinism replay hashes. |
| bottleneck | None of the three tested step-300 running-speed handovers survived; alternative phases, timings, and state-conditioned handovers remain untested. The candidate holds `expert` and omits the slow third. |
| next step | Close Experiment 003 phase A. Phase B runs the same T1 task through a tracker-following fine-tuning runtime warm-started at the expert; no cycle-3 prompt is prepared. |

## Candidate provenance

| item | value |
|---|---|
| designer | `sol-designer-20260905-e003d2`; launcher requested read-only; observed reads are audit-log-only, not OS-enforced isolation |
| designer run directory | `.orchestration/sol-runs/20260905T185111Z-78f29dc3-9fba-4f26-a393-b86bc71f4349` |
| files read by designer | audit log records `cycles/cycle_2/designer_prompt.md` and `cycles/cycle_1/report_1.md`; this is not an OS-enforced read allowlist |
| source | `.orchestration/oracles/cycle_2_candidate.json` |
| frozen copy | `cycles/cycle_2/oracle_2.json` |
| raw file SHA-256 | `f1cbc2784783d4f34312b036a911c536f8406aa3da1ffcf52f35f4927808173a` for both paths; `cmp` matched |
| canonical oracle SHA-256 | `5868d09ea71a4fcd5f1be46ef92fa36c83ac7e3867222cecb9fa7f3000e835bf` |
| contract validation | Pass; oracle ID `cycle_2_candidate`; no oracle edit |
| precondition | Clean `19e73d9636aaac5be231fa32d5b7e9747d0021ff` |

## Fable steering

Source: Fable, the orchestrating agent. No human steering was supplied. Required text, recorded verbatim:

> Both cycle-0 switching arms and the cycle-1 staged candidate fell in 20 of 20 episodes, each time 22 to 57 steps after the expert-to-medium switch at step 300, regardless of dwell, velocity gating, or recovery. Treat the switch at running speed as the suspect. ... Either design a program whose first switch happens only under a state condition you can justify from the diagnostics ..., or conclude that no switch from the expert at its running speed is survivable with this library and return the program that best serves the task without such a switch, stating the infeasibility plainly. Falls first, speed error second.

### Correction recorded 2026-09-05 (SCI-001)

The historical steering above is preserved verbatim, but its evidence summary
is inaccurate. The cycle-0 `playback` arm switched `expert` to `simple` and
fell `20`-`25` steps later; the cycle-0 `handwritten` arm switched `expert` to
`medium` and fell `23`-`60` steps later; only the cycle-1 candidate has the
`22`-`57` range. The cycle-2 designer worked from that inaccurate steering
summary.

## Designer rationale and bounded handover statement

Review-bounded rendering of the designer rationale; the original final response
remains byte-hash-bound in `designer_provenance.json`:

> Hold `expert` throughout: the zero-fall requirement takes precedence over speed error.
>
> All three switching candidates fell in 20/20 episodes; every cycle-1 scheduled switch occurred at 4.80–5.89 m/s and preceded a fall.
>
> None of the three tested step-300 running-speed handovers survived; alternative phases, timings, and state-conditioned handovers remain untested.
>
> Falsifier: under the fixed evaluation, this hold falls, or a library-valid running-speed switch achieves zero falls and lower schedule MAE.
>
> An admitted deceleration behavior with entry coverage at expert running speeds and a validated safe handoff to `medium` or `simple` is one untested next direction.

The returned program has one `expert` state and no transitions.

## Evaluation result

| metric | cycle-2 candidate |
|---|---:|
| episodes / horizon | `20 / 1,000` steps |
| median mean absolute speed error | `2.0741721755 m/s` |
| falls | `0/20` |
| first fall boundary | None in all episodes |
| median controller switches | `0` |
| median behavior time | expert `1,000`; medium `0`; simple `0` steps |
| slow-third fraction `[300,600)` | expert `1`; medium `0`; simple `0` in every episode |
| median descriptive stock task return | `10,648.5529837515` |

- The candidate is behaviorally identical to cycle 0 `single_fast`: both execute the deterministic expert actor for every control step without transitions.
- All comparable per-episode outcome fields matched `single_fast` byte-for-byte for `20/20` seeds: seed, fall, first-fall boundary, observed steps, speed MAE, switch count, task return, and behavior-time counts.
- The complete serialized rows intentionally differ in oracle provenance, trace path, size, and hash, wall time, and cycle-2 diagnostic fields. After excluding oracle/state-name metadata, all `20/20` per-step behavioral trace projections also matched byte-for-byte.
- The candidate passed the never-fall requirement but matched `single_fast` rather than solving the speed profile. It spent the entire requested slow third in `expert`.

## Receipts

| artifact | SHA-256 / result |
|---|---|
| runtime fingerprint | `186fd2f4aa6b9ed9a0eb3de73faa0d2bbcb392a4fe52f992b6b443fbf33d2621`; unchanged from cycles 0 and 1 |
| `report_2.json` | `e677b9f3c3daabdb13648a18981c39b47bc74fa2e6908e7fe1af9d73674e194c` |
| `report_2.md` | corrected render `c071106fe3bbfc050ba1fdcba234a1da32f0d43df365851e5360c54b102420b6` |
| `scientific_receipt_v2.json` | `c6dab6cb6e5a9b524b947ef11aec2e7ae64d4641f4e16828c9fd75b43a4eda7f`; deterministic scientific chain |
| `telemetry_v1.json` | `b22fdfa7a8fb23e38811c5fb6935e5092541d8c69cc45fe74fde8b355594c077`; operational timestamps, host fields, wall times, and receipt bindings |
| `designer_provenance.json` | `2b9ee3b2b53f6cdb3a64585da759505155c6885c31f0a0f4a71fdbe7a764e9c3`; sanitized audit-log-only receipt |
| ignored trace index | `a210e04509a1ac2266981c1ea0af1407ce5059d72d633d6ba62daf9970bdf775`; `20` entries; cycle directory `13M` |
| determinism replay | `20/20` hashes matched; `3.3180 s` inside the report |
| evaluation wall time | `6.8207 s` inside the report; `7.30 s` process wall including startup and writes |

## Validation

| check | result |
|---|---|
| independent artifact audit | Pass: canonical report, exact frozen seeds, all `20` indexed trace bytes and hashes, runtime binding, and byte-exact outcome/behavior projections against `single_fast` |
| focused harness tests | `14 passed` in `0.59 s` process wall; includes three-cycle carry-forward and malformed prior-history rejection |
| Ruff lint / format | Pass; `252` files already formatted |
| final `git diff --check` | Pass in `0.01 s` |
| builder wall time | `13m` from launcher acquisition through final validation and handoff |

## Claim ceiling

This is an exploratory controller-switching cycle on the frozen plain `Humanoid-v5` runtime. Controller switching stands in for tracker following. The result supports no oracle-quality, generalization, frozen-tracker, reference-following, task-reward, naturalness, robustness, or humanoid-competence claim.
