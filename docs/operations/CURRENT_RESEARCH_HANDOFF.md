# Current research handoff

| status | current truth |
|---|---|
| progress | Builder 03A2 verified the lease, human gate, clean `main`, and the byte-exact 20-path WIP preservation on `wip/tqc-v2-attempt-supervisor` at `5bdae45`. No implementation or payload load occurred. |
| bottleneck | The mandatory untouched-main full suite failed: `44 failed, 1014 passed, 2 skipped in 108.12s`. The managed builder sandbox denies AF_UNIX/AF_INET socket binds, and frozen runtime checks observed CPU model `arm` instead of `Apple M5 Max`. The packet requires a stop on any pre-change suite failure. |
| next step | Reproduce `.venv/bin/python -m pytest -q -p no:cacheprovider` on unchanged `main` in the intended host execution context. If it passes, relaunch `TASK-20260904-03A2`; if it still fails, repair the baseline in a separate reviewed slice before relaunch. Do not train or load the external checkpoint. |

- Updated: `2026-09-04T16:58Z`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Branch: `main`
- Baseline HEAD before orchestration: `336ded931334475a3b64384f1257e6d1e7d0e776`. Fable commits on `main`: `4a0976d` (audit and ADRs), then the builder stop record. WIP branch: `wip/tqc-v2-attempt-supervisor` at `5bdae45`.
- Control owner: Fable session `807bcdb2-462c-4ea8-803a-1e4b41259e12`, lease owner `fable-395e7d0f-be34-489e-944e-bbfa673a1eea`
- Fable lease: `CLAIMED` by the Fable owner while Fable works; scope `docs/strategy,docs/operations,docs/decisions,.orchestration/task-packets,artifacts/external`; released at each clean handoff
- Active write worker: none at this update; `sol-builder-20260904-03a2` launches next
- Takeover authorization: disabled; the heartbeat may only report
- Human gates: all three startup gates answered; see "Questions for Samuel" below

## Active read-only Sol workers

| owner | role | packet | run directory | screen | runner PID |
|---|---|---|---|---|---|
| `sol-review-sci-20260904-01` | scientific reviewer | `.orchestration/task-packets/TASK-20260904-01-strategy-scientific-review.md` | `.orchestration/sol-runs/20260904T162138Z-91fdd351-b579-46f8-b317-402f78bd2907` | `hh-sol-5538c277-e542-4505-8c92-787b4798b0d8` | `33469` |
| `sol-review-adv-20260904-02` | adversarial reviewer | `.orchestration/task-packets/TASK-20260904-02-strategy-adversarial-review.md` | `.orchestration/sol-runs/20260904T162140Z-53837dc0-a9f8-4bf5-97f7-ae7e593a60b4` | `hh-sol-f4c09dad-bbe5-40f5-8587-99155b3b88d9` | `33916` |
| `sol-design-reward-20260904-04` | research designer | `.orchestration/task-packets/TASK-20260904-04-reward-track-design-survey.md` | `.orchestration/sol-runs/20260904T163715Z-ddddde81-0085-49fd-8714-0d512c9d39a0` | `hh-sol-342b73b1-463f-4fd0-9f2d-618598681f8f` | `37865` |
| `sol-design-tracker-20260904-05` | research designer | `.orchestration/task-packets/TASK-20260904-05-tracker-track-design-survey.md` | `.orchestration/sol-runs/20260904T164217Z-62e441c4-8938-4986-b267-cb0f8d21ce80` | `hh-sol-25fc9414-83ae-42f7-929f-a3f3c2e5d37d` | `47373` |

Launched `2026-09-04 16:21Z`. Poll with `./scripts/start-sol-worker status RUN_DIR`;
read only `final.txt` and `result.json`, never stream `events.jsonl`. Codex
thread IDs observed at `16:25Z`: scientific `01a06d39-cb22-7af2-a89f-88c80845f1d3`,
adversarial `01a06d39-d229-7b12-81ec-cad50a5a3a90`.

State at `2026-09-04T16:46Z`: reviews `01` and `02` are terminal `SUCCEEDED`, both
`GO-WITH-FIXES` with no P0 (scientific `4` P1, adversarial `9` P1 and `1` P2);
all P1 fixes are folded into ADR 0005, the strategy, and the split builder
packets. Survey `04` running since `16:37Z`; survey `05` running since
`16:42Z`. The original packet `TASK-20260904-03` is superseded by `03A` (this
launch) and a later `03B` (external evaluator adapter, 20-reset screen, replay
bundle).

## Questions for Samuel

| id | question | answer line to add here |
|---|---|---|
| Q1 | May the builder import the public Farama `Humanoid-v5` TQC expert through the data-only path in ADR 0005 (pinned SHA-256 bytes, `torch.load(weights_only=True)` tensors only, JSON-only metadata, no `TQC.load`)? The E1 note currently forbids loading uploaded SB3 or Torch objects. If no: authorize up to `3` receipted local TQC attempts of at most `3 h` CPU each. | `external-checkpoint-import: approved YYYY-MM-DD` or `external-checkpoint-import: denied; local-attempt-budget: N` |
| Q2 | May a reward-only harness loop on stock `Humanoid-v5` (route D in the strategy) run as a parallel second track while the tracker is admitted? It reorders `LG-11`. | `reward-first-track: approved` or `denied` |
| veto | The builder will move the uncommitted TQC-v2 files byte-exact to branch `wip/tqc-v2-attempt-supervisor` and remove them from the `main` working tree. | `wip-branch-move: vetoed` stops it |

Answers recorded `2026-09-04T16:31Z` from Samuel's message in the Fable session:

- `external-checkpoint-import: approved 2026-09-04`
- `reward-first-track: approved`
- `wip-branch-move: approved`
- Samuel ran `/status` and confirmed `claude-fable-5-1`; the identity gate is
  human-confirmed for this session.
- Standing instruction: do not ask about routine route decisions or worker
  launches; run Sol workers directly, delegate token-heavy work to them, and
  keep running autonomously. The contract's hard gates (real 1M-step attempt,
  PRAXIST campaign, formal confirmatory study, paid API usage, push or
  publication) still need explicit authorization.

## Claim boundary

| layer | current statement |
|---|---|
| research target | An LLM-guided harness revises a reference-composition oracle `O_k` and task reward `r_k`, using protected rollout evidence and optional human steering. |
| implemented capability | Typed oracle artifacts; a validated linear phase-window automaton; Gymnasium adapter; deterministic trace/metric contracts; CLI; read-only evidence UI; research-source ledger. Uncommitted TQC-v2 WIP is present, passes its focused tests, and is parked by ADR 0005. |
| measured evidence | Interface and regression checks; one non-admitted falling tracker exploration; one reviewed offline numeric-reference sensitivity probe; one 100k resource calibration; `tests/experiments` `822 passed` on the WIP tree; hash-verified local copy of a public expert artifact (an artifact record, not behavior). |
| not demonstrated | Stable Humanoid tracking; causal policy use of reference windows; better transitions; recovery; oracle improvement; reward improvement; cross-MDP generalization; an autonomous closed research loop. |

## Scientific dependency chain

```text
Experiment 001             Experiment 002                 Experiment 003+
admit stable tracker  -->  prove causal reference use --> compare oracle arms
   ROUTE: ADR 0005              BLOCKED                   NOT AUTHORIZED
```

- A time-indexed reward is not proof that actor and critic consumed the
  immutable reference clock and window.
- Oracle quality cannot be inferred from the current falling video.
- Reference generation and reference composition are separate problems. This
  project currently owns composition.
- Task-reward generation is the approved parallel Family-B track (ADR 0006).
  It stays fixed inside every oracle-only comparison.
- The public expert generated the registered Minari expert clips. A tracker
  built on it can follow its own rollout while ignoring the reference; gate E3
  and non-self reference gaits are mandatory.

## Route decision

Recorded in `docs/strategy/RESEARCH_STRATEGY.md`, section "2026-09-04 startup
audit", and `docs/decisions/0005_public_expert_base_controller.md`.

| route | verdict |
|---|---|
| Finish the TQC-v2 one-attempt supervisor | parked: `8` open P1 lifecycle findings and `4,499` uncommitted lines guarding a `27`-minute job |
| Public expert import, data-only, receipted local fallback | recommended; scientific review `GO-WITH-FIXES` folded; adversarial review pending |
| Reward-first loop on stock `Humanoid-v5` | approved parallel track (ADR 0006); design survey running |

## Working-tree state

- Start state was clean at `main` HEAD
  `48955dfb299498d1893e6dfffe9facb87a4192a5`; the three-dot comparison and
  commit `5bdae45` each list exactly the expected 20 preserved paths.
- `TASK-20260904-03A2` changed-files list before stopping:
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`: required three-row stop
    status, receipt, changed-files list, and exact Fable resume action only.
- Pre-change receipt: `.venv/bin/python -m pytest -q -p no:cacheprovider`
  produced `44 failed, 1014 passed, 2 skipped in 108.12s`. Representative hard
  failures were `PermissionError: [Errno 1] Operation not permitted` while
  binding multiprocessing/UI sockets and `observed preflight cpu_model differs`
  (`arm` observed; `Apple M5 Max` frozen).
- No registration files, importer, tests, strict actor NPZ, import receipt,
  equivalence receipt, checkpoint load, training run, or behavior evaluation
  were created. The artifact hashes were not rechecked after the suite stop.

## Current no-go findings

These findings were observed against the uncommitted TQC-v2 working tree. They
remain open on the parked code and no longer sit on the route, but must be
retested if that code ever returns.

| id | blocker |
|---|---|
| TQC-P1-01 | The one-attempt rule is directory-local and can be bypassed with another fresh directory. |
| TQC-P1-02 | A failure after process start can escape complete cleanup and can misreport whether a worker started. |
| TQC-P1-03 | Process-group signal authority can be inferred from the OS before worker self-report is authenticated. |
| TQC-P1-04 | Authenticated worker failures before manifest acknowledgement can be downgraded to generic integrity failures. |
| TQC-P1-05 | Signal and terminal-receipt ownership is not safe across every public launch path. |
| TQC-P1-06 | A transient macOS `EPERM` group probe can false-reject cleanup instead of remaining indeterminate within the deadline. |
| TQC-P1-07 | The training-to-persistence semantic bridge for policy, entropy, and optimizer state is unresolved. |
| TQC-P1-08 | Success does not yet revalidate every retained preflight, manifest, reservation, and worker receipt. |

Additional constraint:

- A production-ID canary would consume the sole attempt. Any canary must use a
  separate, explicitly non-authorizing identity. This constraint lapses with
  the one-attempt rule if ADR 0005 survives review.

## Orchestration state

| component | observed state |
|---|---|
| Claude Code | `2.1.260`; session `807bcdb2…` recorded `claude-fable-5-1` at startup; hooks accepted every tool call |
| Fable policy | Exact model requested at launch; `switchModelsOnFlag=false`; Samuel's `/status` pending |
| session guard | Project hooks deny requested non-Fable switches and all Claude tool use when the recorded model is not Fable or active effort is not `max` |
| Codex | Official standalone CLI at `~/.local/bin/codex`; `.codex/config.toml` requests `gpt-5.6-sol` at `max` with `multi_agent=false` |
| worker path | `scripts/start-sol-worker` made its first two real launches at `16:21Z`, both read-only, both in detached screen sessions with compact status |
| write control | Fable held the strategy lease for the audit and released it at the end of the startup turn; the recorded scope is an audit boundary |
| continuity | The 30-minute heartbeat is coordination-only; no takeover record exists |

## Resume order

1. Run `./scripts/orchestration-doctor` for the no-usage local checks.
2. Run `./scripts/start-fable-orchestrator`; authorize its one live identity
   probe when prompted.
3. Fable reads `CLAUDE.md`, the master prompt, this handoff, the strategy
   audit section, and ADR 0005. The raw private sources need re-reading only if
   a decision depends on disputed wording.
4. Samuel runs `/status`; Fable checks the latest session-start model event.
5. Fable runs `./scripts/start-sol-worker status RUN_DIR` for both review runs
   and reads each `final.txt`.
6. Under a fresh Fable lease, Fable records accepted findings in the strategy,
   ADR 0005, and this file, then releases the lease.
7. If Samuel's Q1 answer line is present and the reviews say `GO` or
   `GO-WITH-FIXES` with fixes folded into the packet, launch:
   `./scripts/start-sol-worker launch write sol-builder-20260904-03 builder src/oracle_composition,tests,experiments/bootstrap_tqc_humanoid,research/source_controllers,docs/experiments,README.md,experiments/README.md,pyproject.toml,artifacts/bootstrap_tqc_humanoid .orchestration/task-packets/TASK-20260904-03-import-public-expert-base-controller.md`
8. Update this file at least every 30 minutes during active work and at every
   control transfer.

Fable resume: on unchanged `main`, run `.venv/bin/python -m pytest -q -p
no:cacheprovider` in the intended host execution context with socket binding
available; relaunch `TASK-20260904-03A2` only after that exact suite passes.

## Handoff update contract

- Keep the three-row table first.
- Replace, do not append to, its current statements.
- Separate requests, implementation, and measurements.
- Name changed files, test receipts, remaining blockers, active worker/thread
  IDs, and the next bounded action.
- Never convert a green test, UI launch, or zero-action smoke test into a
  behavioral claim.

## Git-state rule observed 2026-09-04

The Codex `workspace-write` sandbox denies writes under `.git`. Sol workers
cannot branch, stage, commit, stash, or create worktrees. Fable performs every
git state change under its own lease after review. Builder 03A stopped at that
boundary at `16:52Z` with all preconditions verified; Fable completed the WIP
preservation at `2026-09-04T16:58Z` with `artifacts/bootstrap_tqc_humanoid/wip_preservation_fable_verification.txt`
as the byte-level receipt.
