# Current research handoff

| status | current truth |
|---|---|
| progress | B0FIX2 landed the unresolved venv launch, Seatbelt profile v2, interpreter identity binding, and categorical exit diagnostics; the host now shows the worker starting. Fable's host diagnosis found the last gap: Python cannot list `src` under the profile, so the package resolves as a namespace shell from site-packages; three directory-listing grants fix it. |
| bottleneck | Nested Seatbelt remains unavailable inside Codex, so 14 OS canaries are explicitly unverified here; profile v2 still needs Fable's host rerun. No tracker is admitted, and causal reference use remains unproved. |
| next step | Run `B0FIX3` (three directory grants, profile v3) as the writer, verify the `20` canaries on the host, commit B0 as `Slice B0`, launch its two reviews read-only, and run `03A2FIX` in parallel. Do not launch any 1M-step attempt. |

- Updated: `2026-09-04T21:42Z`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Branch: `main`
- Baseline HEAD before orchestration: `336ded931334475a3b64384f1257e6d1e7d0e776`. Fable commits on `main`: `4a0976d` (audit and ADRs), then the builder stop record. WIP branch: `wip/tqc-v2-attempt-supervisor` at `5bdae45`.
- Control owner: Fable session `807bcdb2-462c-4ea8-803a-1e4b41259e12`, lease owner `fable-395e7d0f-be34-489e-944e-bbfa673a1eea`
- Fable lease: `CLAIMED` by the Fable owner while Fable works; scope `docs/strategy,docs/operations,docs/decisions,.orchestration/task-packets,artifacts/external`; released at each clean handoff
- Active write worker: `sol-builder-20260904-03a2r`, completed slice ready for Fable review and control transfer
- Takeover authorization: disabled; the heartbeat may only report
- Human gates: all three startup gates answered; see "Questions for Samuel" below

## Active read-only Sol workers

| owner | role | packet | run directory | screen | runner PID |
|---|---|---|---|---|---|
| `sol-review-sci-20260904-01` | scientific reviewer | `.orchestration/task-packets/TASK-20260904-01-strategy-scientific-review.md` | `.orchestration/sol-runs/20260904T162138Z-91fdd351-b579-46f8-b317-402f78bd2907` | `hh-sol-5538c277-e542-4505-8c92-787b4798b0d8` | `33469` |
| `sol-review-adv-20260904-02` | adversarial reviewer | `.orchestration/task-packets/TASK-20260904-02-strategy-adversarial-review.md` | `.orchestration/sol-runs/20260904T162140Z-53837dc0-a9f8-4bf5-97f7-ae7e593a60b4` | `hh-sol-f4c09dad-bbe5-40f5-8587-99155b3b88d9` | `33916` |
| `sol-design-reward-20260904-04` | research designer | `.orchestration/task-packets/TASK-20260904-04-reward-track-design-survey.md` | `.orchestration/sol-runs/20260904T163715Z-ddddde81-0085-49fd-8714-0d512c9d39a0` | `hh-sol-342b73b1-463f-4fd0-9f2d-618598681f8f` | `37865` |
| `sol-builder-20260904-03a` (write) | builder | `TASK-20260904-03A` | `.orchestration/sol-runs/20260904T164729Z-273705b5-fd1b-49d6-9495-98e63f907b01` | stopped at the `.git` boundary; superseded | `50887` |
| `sol-builder-20260904-03a2` (write, launch 1) | builder | `TASK-20260904-03A2` | `.orchestration/sol-runs/20260904T165851Z-cde09967-82e9-425c-9b16-91292ac226f5` | stopped: two-dot diff precondition | `58734` |
| `sol-builder-20260904-03a2` (write, launch 2) | builder | `TASK-20260904-03A2` | `.orchestration/sol-runs/20260904T170134Z-f48b6fc2-905b-4323-87f3-8b88246888c4` | stopped: in-sandbox suite failures | `59876` |
| `sol-builder-20260904-03a2` (write, launch 3) | builder | `TASK-20260904-03A2` | `.orchestration/sol-runs/20260904T171428Z-2d37493e-5ef0-455c-8c58-5f9878a2074b` | partial slice in the working tree; stopped at `process safe globals refused` | `62996` |
| `sol-builder-20260904-03a2r` (write) | builder | `TASK-20260904-03A2R` | `.orchestration/sol-runs/20260904T174545Z-7b364f7f-b3bf-426d-8023-17389e1f0c12` | `SUCCEEDED`: slice complete, thread `01a06d86-cd17-7ee2-b6de-c7d3d07576f9` | `67383` |
| `sol-review-sci-20260904-06` | scientific reviewer of `1d6b461` | `TASK-20260904-06` | `.orchestration/sol-runs/20260904T181613Z-721ea785-8dc3-4d8f-9eb6-84b4672d1fb3` | `hh-sol-869272fb-cc6c-480d-9b0a-0b3eece424a9` | `74477` |
| `sol-review-adv-20260904-07` | adversarial reviewer of `1d6b461` | `TASK-20260904-07` | `.orchestration/sol-runs/20260904T181614Z-5bb41e36-fc24-49b0-8fb3-a73409210b5a` | `hh-sol-d7eb4fef-93a3-49a5-9161-140a7d1672a7` | `74544` |
| `sol-builder-20260904-b0` (write) | builder | `TASK-20260904-B0` | `.orchestration/sol-runs/20260904T181636Z-8fcc968b-0d9a-4901-ba7f-3a389fc522af` | `SUCCEEDED`; slice uncommitted pending B0FIX; thread `01a06da3-08f9-7a92-b01e-eb2bacdb3502` | `76786` |
| `sol-builder-20260904-b0fix` (write) | builder | `TASK-20260904-B0FIX` | `.orchestration/sol-runs/20260904T202141Z-260b15ed-0b8b-47df-a17b-800bc2267504` | `SUCCEEDED`; thread `01a06e15-8f53-7452-ad7a-00a9d3688a14` | `46776` |
| `sol-review-adv-20260904-08` | robustness reviewer of `1d6b461` | `TASK-REVIEW-ROBUSTNESS-GENERIC` | `.orchestration/sol-runs/20260904T202141Z-fe0d2655-8bd5-4363-8445-abf0e2d51bda` | `SUCCEEDED`; `ACCEPT-WITH-REPAIRS`; thread `01a06e15-8faa-7453-8319-d724dc56ff5f` | `46844` |
| `sol-builder-20260904-b0fix2` (write) | builder | `TASK-20260904-B0FIX2` | `.orchestration/sol-runs/20260904T210814Z-f73f6df5-757f-4fab-b3aa-17f8c660935b` | `SUCCEEDED`; thread `01a06e40-2e02-7540-843e-b7b908f1ae37` | `60764` |
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
- `TASK-20260904-B0` tracked or trackable changed-files list:
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/DECISION_RULE.md`.
  - `experiments/family_b_target_speed_v1/configs/family_b_target_speed_v1.study.json`.
  - `experiments/family_b_target_speed_v1/candidates/stock_r0.py`.
  - `experiments/family_b_target_speed_v1/candidates/manual_target_speed_v1.py`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`.
  - `experiments/family_b_target_speed_v1/receipts/builder_synthetic_scale.json`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`.
  - `src/oracle_composition/rewards/__init__.py`.
  - `src/oracle_composition/rewards/contract.py`.
  - `src/oracle_composition/rewards/stock_humanoid.py`.
  - `src/oracle_composition/rewards/static_validation.py`.
  - `src/oracle_composition/rewards/scale_calibration.py`.
  - `src/oracle_composition/rewards/sandbox.py`.
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `src/oracle_composition/experiments/reward_target_speed_evaluator.py`.
  - `src/oracle_composition/experiments/reward_target_speed_manifest.py`.
  - `tests/rewards/test_contract.py`.
  - `tests/rewards/test_stock_humanoid.py`.
  - `tests/rewards/test_static_validation.py`.
  - `tests/rewards/test_scale_calibration.py`.
  - `tests/rewards/test_sandbox.py`.
  - `tests/experiments/test_reward_target_speed_evaluator.py`.
  - `tests/experiments/test_reward_target_speed_manifest.py`.
  - `artifacts/family_b_target_speed_v1/sandbox_baseline_failures.txt` (ignored
    in-sandbox baseline receipt).
- `TASK-20260904-B0FIX` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/sandbox.py`.
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    canonical SHA-256
    `b0cbe91ac54199906c4e68a785bb8af03cc0d9c904997a1a16845fdfa8eccacc`;
    file SHA-256
    `eb340a34e22e150506251e7d6322229b30d84f06cdf355dbd97ad92fa348d891`.
  - `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json`
    (unchanged historical diagnostic): canonical SHA-256
    `62c2529fd0430b0d9bd9fe7c0920015535961aa9771874aa65f0cecfad161190`;
    file SHA-256
    `1f29d913baabc1bcab49c1ed27417477c8950db3e8dba5a6bda77a962e9ad05d`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`:
    canonical SHA-256
    `0f14245b8c63168e9f7d3d760a15bc2d21846abeeb2858c9e3bcaeaa9effab7c`;
    file SHA-256
    `8f216c6584610ef127909daba4309d38c6a53223114f0398d37fdf29c7203d0a`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: `87 passed, 7 skipped` focused; full suite `44 failed, 1140
    passed, 9 skipped`, with the exact 44 IDs in
    `artifacts/family_b_target_speed_v1/sandbox_baseline_failures.txt`; Ruff
    lint and format checks pass.
- `TASK-20260904-B0FIX2` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/sandbox.py`.
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    canonical SHA-256
    `22aa4e86fcafc52067bee2c8b34ec539b4f2ea2ba768807c12e8fb59b936bac6`;
    file SHA-256
    `e9349a74bdc933bef9a48f00fca3ad44dff342693246f943f7597e11a740fff6`;
    profile SHA-256
    `7dcef9b4e429752695700817d0748560f6832e2d8b6b1f6f4ec253f2a423a816`.
  - Interpreter launch path `.venv/bin/python` and resolved uv binary each
    hash to
    `7710b0490e6af648676d7ad163fa1d54bffb75d8505162609a1c3c9b76e1929d`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`
    was refreshed only for source identity: canonical SHA-256
    `0ef5ce4bc3205c0b4d6abf84f60d9e0ab6028d265a28cca8f8c17cdb6a283063`;
    file SHA-256
    `164bdddf57a0f33e43cb57a391223190ef70a42b09c8f0312f008838bedc9d78`.
  - `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json`
    is unchanged historical profile-v1 evidence: canonical SHA-256
    `94b576cf6372902612e7cf68c2f9e5fbaf26bec707d5ebd52dc92aa79a68c786`;
    file SHA-256
    `a17291848ed15f210294f65849588456aa565bccde7715c39d5d160b0b9aeb4d`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: sandbox tests `22 passed, 7 skipped` with exact reason
    `sandbox-exec: sandbox_apply: Operation not permitted`; full suite `44
    failed, 1143 passed, 9 skipped`, with the exact saved 44-test baseline;
    Ruff lint and format checks pass. No training or behavioral evaluation ran.
- Local ignored artifacts created by the slice:
  - `artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json`: SHA-256
    `ce90c312f7c222847a936edd2d964d386bab01d1acb0fd924de3a8db951430cb`,
    11,308 bytes.
  - `artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz`:
    SHA-256
    `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`,
    618,674 bytes; actor-state fingerprint
    `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`.
  - `artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_v1.json`:
    SHA-256
    `d57eedf35ded4dc5e1f9e6f1cb04c9a67f5db1b491d1825ae6a2196099fc6d32`,
    2,429 bytes.
- Metadata registration receipt: SHA-256
  `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf`,
  9,395 bytes. LFS siblings are labeled `lfs.sha256`; small files are labeled
  `git-blob-sha1` from `blobId`; each LFS API record retains its `lfs` sub-object.
- Safe-globals receipt: 75 sorted qualified names, SHA-256
  `7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9`;
  only `builtins`, `traceback`, `collections`, `torch`, and `torch.*` modules;
  unchanged before/after the one weights-only load.
- Test receipts:
  - new focused files: `39 passed in 2.37s`;
  - extended focused set including committed NPZ/equivalence regressions: `81
    passed, 1 skipped in 4.73s`;
  - full suite: `44 failed, 1053 passed, 2 skipped in 112.82s`; its sorted
    failing-node set equals the 44-node baseline at
    `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures.txt` (SHA-256
    `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345`),
    with no new failures;
  - receipt/NPZ/index revalidation: registration valid, import receipt valid,
    strict NPZ valid, equivalence receipt valid, Git-index policy passed;
  - `.venv/bin/python -m ruff check .`: all checks passed;
    `.venv/bin/python -m ruff format --check .`: 191 files already formatted.
- No training, environment rollout, 20-reset screen, or behavior evaluation ran.

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

Fable resume: rerun the 20 B0 Seatbelt canaries outside the builder sandbox,
resolve the exact 1 GiB `RLIMIT_AS` startup failure if it repeats, then review
the scope-only diff and three B0 receipts and commit; do not run training or
credit fixed-control parity as behavior.

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

## Reward-track design adopted `2026-09-04T17:10Z`

Survey `TASK-20260904-04` (thread `01a06d48-1482-73f2-800a-a3bb69d9de5e`,
run `.orchestration/sol-runs/20260904T163715Z-ddddde81-0085-49fd-8714-0d512c9d39a0`)
delivered the `family-b-target-speed-v1` design. ADR 0006 records it. Builder
packet `TASK-20260904-B0` implements the no-training first slice. Review
packets `06` and `07` are written for the 03A2 diff.

## Baseline suite receipt `2026-09-04T17:13Z`

`.venv/bin/python -m pytest -q -p no:cacheprovider` on clean `main` at
`48955df`, outside the Sol sandbox: `1060 passed, 2 warnings in 122.17s`
(`17:08:13Z` to `17:10:16Z`). Software behavior only. Inside the Sol sandbox
the same tree shows `44 failed` for environment reasons (denied socket binds,
CPU fingerprint `arm`); builders record their own in-sandbox baseline and must
not add failures.

## Tracker-track design adopted `2026-09-04T17:13Z`

Survey `TASK-20260904-05` (thread `01a06d4c-b071-7493-863b-fbcdaafa6c5d`,
run `.orchestration/sol-runs/20260904T164217Z-62e441c4-8938-4986-b267-cb0f8d21ce80`)
delivered the same-runtime reference, Tier-D, E3, residual-PPO E4, and E5
designs. ADR 0005 records the chain. Packets `03A3` and `03B` implement the
first two slices; packets for E4 and E5 follow their reviews.

## Builder 03A2 diagnostic `2026-09-04T17:45Z`

Fable ran the worker's stages in a separate process on the pinned expert
`policy.pth`: resource limits applied (macOS reports no finite `RLIMIT_AS`),
then category `process safe globals refused`. A fresh interpreter on Torch
`2.14.0` reports `75` pre-registered safe globals (builtin exception classes,
`traceback.FrameSummary`, Torch internals). The corrected invariant is
"no additions during the load and no entry outside `builtins`, `traceback`,
`collections`, or `torch`", recorded by count and SHA-256 in the receipt.
Packet `TASK-20260904-03A2R` carries the fix and the remaining steps. The seven
uncommitted code files from launch 3 stay in the working tree for it.

## External actor import receipts `2026-09-04T18:13Z`

Builder `03A2R` completed the slice. Local ignored artifacts under
`artifacts/bootstrap_tqc_humanoid/`: registration receipt SHA-256
`5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf`; import
receipt `ce90c312f7c222847a936edd2d964d386bab01d1acb0fd924de3a8db951430cb`;
strict actor NPZ `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`
(`618,674` bytes); equivalence receipt
`d57eedf35ded4dc5e1f9e6f1cb04c9a67f5db1b491d1825ae6a2196099fc6d32`;
actor-state fingerprint
`3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`; safe
globals `75` entries, hash `7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9`.
In-sandbox suite: `44 failed, 1053 passed, 2 skipped`, failures identical to
the frozen baseline; new focused tests `39 passed`; ruff lint and format
passed. Evidence class `external_base_import`, an `interface_check`. Nothing
behavioral ran.

## Review round on `1d6b461` and B0 host checks `2026-09-04T20:21Z`

- Scientific review `06` (thread `01a06da2-b496-7cb2-b89a-c87725723b53`): `ACCEPT-WITH-REPAIRS`, `0` P0, `3` P1 (`SCI-03A2-01` nominal payload fingerprint; `-02` sorted safe-globals comparison; `-03` vacuous negatives). Repair packet `TASK-20260904-03A2FIX` also carries the storage-interval overlap check found by the adversarial probe.
- Adversarial review `07` (thread `01a06da2-b54c-70a1-8ffb-c2a2f033d1e6`): `turn.failed`, flagged by the Codex cyber-safety filter after constructing one overlap probe. Future adversarial packets are phrased as defensive robustness reviews without payload construction (`TASK-REVIEW-ROBUSTNESS-GENERIC`), driven by `.orchestration/review-target-adv.txt`. A matching `TASK-REVIEW-SCI-GENERIC` reads `.orchestration/review-target-sci.txt`.
- B0 host checks: Seatbelt bootstrap passes on the host; `RLIMIT_AS 1 GiB` raises `ValueError` on macOS; `14` OS canaries fail in pre-exec, `6` protocol canaries pass; host receipt `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json` (canonical SHA-256 `62c2529fd0430b0d9bd9fe7c0920015535961aa9771874aa65f0cecfad161190`); outside-sandbox suite `7 failed, 1179 passed`, all seven in `tests/rewards/test_sandbox.py`; ruff clean. Repair packet `TASK-20260904-B0FIX`.

## B0 host diagnosis `2026-09-04T21:08Z`

After B0FIX, the host run of the OS canaries still failed with
`worker pipe closed before a complete response`. Direct execution of one
canary showed (1) `ModuleNotFoundError: oracle_composition` under a permissive
profile because the runner resolves the venv symlink to the base uv
interpreter, which has no venv context under `-I`; and (2) `SIGABRT` at
startup under the v1 profile, fixed only by `(allow file-read-metadata)` plus
`(allow file-read-data (literal "/"))` with data reads under `/Users`. Leave-one-out
over every other top-level area left the worker running. Packet
`TASK-20260904-B0FIX2` carries both repairs. The reward-track worker's exit
code `64` was observed on the host and must be documented by the builder.

## B0 host diagnosis, second round `2026-09-04T21:42Z`

Under profile v2 with the venv interpreter, the worker starts and reports
`ModuleNotFoundError: No module named 'oracle_composition.rewards'`. Inside the
sandbox `oracle_composition` resolves as a namespace package from
`.venv/lib/python3.13/site-packages/oracle_composition/` (the force-included
`_runtime/uv.lock` resource) because listing `src` is a data read the profile
does not grant. Three literal directory grants (`src`,
`src/oracle_composition`, `src/oracle_composition/rewards`) let the regular
package win; the worker then runs to its documented exit paths. Packet
`TASK-20260904-B0FIX3` applies them as profile v3.
