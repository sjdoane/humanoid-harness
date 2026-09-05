# Current research handoff

| status | current truth |
|---|---|
| progress | 03A3 imported the public `medium` and `simple` actors through the hardened importer and passed actual E1 on the real expert bytes for both contenders (`166` focused tests). Reviews of `425d3bc`: scientific `ACCEPT-WITH-REPAIRS` with one provenance P1; robustness `ACCEPT-WITH-REPAIRS` with eight hardening P1 and two P2. Astra acknowledged the interpretation and the B0 transfer. |
| bottleneck | The reward runtime receipt's whole-tree fingerprint breaks on every commit until Astra scopes it (a reward-lane P1). Importer hardening R-01 to R-08 is scheduled after the corpus slice because 03B consumes only the integrity-verified NPZ receipts and does not exercise the importer. No tracker is admitted; causal reference use is unproved. |
| next step | Commit `Slice 03A3` and a separate integration commit, launch its two reviews, run `03B` (three-gait reference corpus, Tier-D certificates, E3 forks, expert development screen) as the writer, then the importer closure slice `03A2FIX2`. Poll the mailbox at each checkpoint. |

- Updated: `2026-09-05T03:13Z`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Branch: `main`
- Baseline HEAD before orchestration: `336ded931334475a3b64384f1257e6d1e7d0e776`. Fable commits on `main`: `4a0976d` (audit and ADRs), then the builder stop record. WIP branch: `wip/tqc-v2-attempt-supervisor` at `5bdae45`.
- Control owner: Fable session `807bcdb2-462c-4ea8-803a-1e4b41259e12`, lease owner `fable-395e7d0f-be34-489e-944e-bbfa673a1eea`
- Fable lease: `CLAIMED` by the Fable owner while Fable works; scope `docs/strategy,docs/operations,docs/decisions,.orchestration/task-packets,artifacts/external`; released at each clean handoff
- Active write worker: `sol-builder-20260904-03a3`, exact lease owner; stopped at the protected Family-B receipt boundary
- Fable resume: inspect the `TASK-20260904-03A3` changed-file and receipt tables below, refresh only the Family-B no-learning runtime receipt under its owning authority, then rerun the full suite and commit after review.
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
| `sol-builder-20260904-b0fix3` (write) | builder | `TASK-20260904-B0FIX3` | `.orchestration/sol-runs/20260904T214238Z-b4e1b91c-2e42-434c-b3e7-776536d48e50` | `SUCCEEDED`; thread `01a06e5f-ad2e-7852-a327-f8342a75cc73` | `73489` |
| `sol-builder-20260904-b0fix4` (write) | builder | `TASK-20260904-B0FIX4` | `.orchestration/sol-runs/20260904T215855Z-02389704-6e96-4274-9b41-2b1a06b5f8ad` | `SUCCEEDED`; thread `01a06e6e-9bf7-79f3-a373-b5ed704cd40b` | `78268` |
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
- `TASK-20260904-03A3` changed-files list and receipts:
  - Preconditions passed at `main` HEAD
    `425d3bcf203929403205a8e8f01c0cc453b60b0b`: the working tree was clean,
    the exact `sol-builder-20260904-03a3` lease was `CLAIMED`, and the 03A2
    expert chain was present. The required pre-change suite was `44 failed,
    1198 passed, 11 skipped`; the sorted baseline is
    `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures_03a3.txt`,
    SHA-256
    `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345`.
  - All 18 present source files across the `medium` and `simple` local copies
    independently matched their API byte counts and `lfs.sha256` or computed
    Git-blob SHA-1 before any load. Both policy files also matched their pinned
    SHA-256 and byte counts.
  - Trackable files:
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_medium/README.md`.
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_medium/RECEIPT.json`.
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_simple/README.md`.
    - `research/source_controllers/farama_minari_humanoid_v5_tqc_simple/RECEIPT.json`.
    - `src/oracle_composition/sources/farama_tqc_sibling_registrations.py`.
    - `src/oracle_composition/sources/external_sb3_actor.py`.
    - `src/oracle_composition/sources/external_payload_policy.py`.
    - `src/oracle_composition/experiments/external_tqc_actor_equivalence.py`.
    - `src/oracle_composition/experiments/external_tqc_initialization_identity.py`.
    - `tests/sources/test_farama_tqc_sibling_registrations.py`.
    - `tests/sources/test_external_sb3_actor_siblings.py`.
    - `tests/sources/test_external_sb3_actor.py`.
    - `tests/experiments/test_external_tqc_initialization_identity.py`.
    - `tests/policy/test_no_external_payload_in_index.py`.
    - `experiments/bootstrap_tqc_humanoid/PUBLIC_EXPERT_IMPORT.md`.
    - `experiments/bootstrap_tqc_humanoid/E1_INITIALIZATION_IDENTITY.md`.
    - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Source registration receipts:

    | variant | repository commit | receipt SHA-256 | bytes |
    |---|---|---|---:|
    | `medium` | `949f7963c1a8964587dca73d48873ad021e168b5` | `68344e980e42543ddd5ca3d164a9c8173f63950d47b4c8c9064e57283fd4e0a1` | 9,401 |
    | `simple` | `39e2954c193fc1352535935a7d71eca9e5745d9b` | `745d82dea8fc4878f70ed58f4d6ff4cd506b0f94c85676dc70e3cffce8283ac5` | 9,401 |

  - Local ignored import artifacts:

    | variant | import receipt SHA-256 | strict NPZ SHA-256 | equivalence receipt SHA-256 |
    |---|---|---|---|
    | `expert` | `b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7` | `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b` | `65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e` |
    | `medium` | `b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172` | `2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04` | `ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487` |
    | `simple` | `bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7` | `b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917` | `604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268` |

  - Actual E1 ignored receipt
    `artifacts/bootstrap_tqc_humanoid/e1_initialization_identity_external_v1.json`:
    SHA-256
    `28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266`,
    11,023 bytes. Both contenders pass bitwise identity on the imported expert;
    authority is `external_pretrained_artifact`, training/environment steps are
    zero, and behavior was not evaluated.
  - Verification: focused import, equivalence, E1, payload-policy, and retained
    fixture tests `166 passed`; source/API/registration/import/NPZ/equivalence/E1
    revalidation passes; Ruff lint and format checks pass for 219 files;
    `git diff --check` passes. Post-change full suite `45 failed, 1221 passed,
    11 skipped`; the 44 baseline IDs remain exact and one additional failure is
    `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
    It is solely the protected Family-B source-tree binding (`7496476d...319c9c`
    recorded versus `ba4faf45...23a13d` observed).
  - Stop boundary: the required post-change baseline gate is not met. The
    Family-B receipt is outside this scope and forbidden, so the builder did
    not refresh it or weaken its validator.
  - Fable resume: review the 03A3 diff and receipts, refresh the Family-B
    no-learning runtime receipt under its owning authority, rerun the full
    suite, and commit only after the additional failure clears.
- `TASK-20260904-03A2FIX` changed-files list and receipts:
  - `src/oracle_composition/experiments/external_tqc_actor_equivalence.py`.
  - `src/oracle_composition/sources/_external_sb3_actor_worker.py`.
  - `src/oracle_composition/sources/external_payload_policy.py`.
  - `src/oracle_composition/sources/external_sb3_actor.py`.
  - `src/oracle_composition/sources/farama_tqc_registration.py`.
  - `tests/policy/test_no_external_payload_in_index.py`.
  - `tests/sources/test_external_sb3_actor.py`.
  - `tests/sources/test_farama_tqc_registration.py`.
  - `experiments/bootstrap_tqc_humanoid/PUBLIC_EXPERT_IMPORT.md`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Regenerated ignored import receipt
    `artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json`: schema
    v2, SHA-256
    `c2379a21079c40928548b593cb86a8fd8eea468787cf31c28481792d3121e80b`,
    12,465 bytes.
  - Regenerated ignored equivalence receipt
    `artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_v1.json`:
    schema v2, SHA-256
    `0d5ec467051e2eebc16a0a186afbe85cc1364afbeab8fcbefa41f088cb656f31`,
    2,687 bytes.
  - Unchanged registration receipt
    `research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json`:
    SHA-256
    `5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf`,
    9,395 bytes.
  - Unchanged actor NPZ
    `artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz`:
    SHA-256
    `60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b`,
    618,674 bytes; unchanged actor-state fingerprint
    `3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a`.
  - Reused sandbox baseline receipt
    `artifacts/bootstrap_tqc_humanoid/sandbox_baseline_failures.txt`: SHA-256
    `c06586e4449fad6b00e6356e6a70e2d75e3476fb8f4fbbf0b2c4e72c547a8345`.
  - Verification: focused source/policy/equivalence tests `173
    passed, 1 skipped`; the skip is `MPS is unavailable`. Full suite `45
    failed, 1197 passed, 11 skipped`: all 44 saved sandbox failures plus only
    `tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
    The mismatch is confined to `source_tree_sha256`; the forbidden reward
    receipt was regenerated by Fable as an integration change (see the provenance correction below). Ruff lint passed and all 212 Python files are
    formatted. No training or behavior evaluation ran.
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
- `TASK-20260904-B0FIX3` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/sandbox.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    canonical SHA-256
    `3cfc64c0868c606cce9b400cf88493f2b9057d7c7dfe668cfd016b6114bf6238`;
    file SHA-256
    `ba1cb6246298d90f7011ed559c1d1829e636b0bf8bf62d82ae4d980146b5a4df`;
    profile SHA-256
    `f18216e5c08a0ac2eac8986f8338b032c84d2de24a231083ae28083bb061f44f`.
  - Profile-delta proof: removing the three literal package-directory rules
    from v3 reproduces the recorded v2 profile SHA-256
    `7dcef9b4e429752695700817d0748560f6832e2d8b6b1f6f4ec253f2a423a816`.
  - Canary verdicts: `source_drift`, `pickle`, `object_dtype`,
    `oversize_frame`, `extra_frame`, and `malformed_response` pass;
    `file_read`, `file_write`, `environment_secret`, `network`,
    `process_creation`, `fork`, `signal`, `tracing`, `repository_read`,
    `timeout`, `memory_abuse`, `stdout_injection`, `private_fd_injection`, and
    `worker_crash` are `not_verified_in_builder_sandbox` with exact error
    `sandbox-exec: sandbox_apply: Operation not permitted`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`:
    canonical SHA-256
    `552e2e39a7438d76af9f7025b6413080d63d791c19be1fa398d8c3a34b257ca0`;
    file SHA-256
    `5cb1d36a64a27a1eb1055af18deb384291de82139048046de61ce1e8fd1b77e5`;
    runtime fingerprint SHA-256
    `d1bf3d270429f1d41d9743f21a6bd090ffadd41e1e500a67b9afeabf5827da50`;
    unchanged eight-step smoke SHA-256
    `048cf64ca8b06f6bd9f00c1b591ba1d0b40fb87620114db6f3a3c367f4218bdc`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: sandbox tests `22 passed, 7 skipped` with exact reason
    `sandbox-exec: sandbox_apply: Operation not permitted`; runtime receipt
    replay `1 passed`; full suite `44 failed, 1143 passed, 9 skipped`, with
    observed failure IDs exactly equal to the saved 44-test baseline; Ruff
    lint and format checks pass. No training or behavioral evaluation ran.
- `TASK-20260904-B0FIX4` repair changed-files subset and receipts:
  - `src/oracle_composition/rewards/_sandbox_worker.py`.
  - `tests/rewards/test_sandbox.py`.
  - `experiments/family_b_target_speed_v1/PROTOCOL.md`.
  - `experiments/family_b_target_speed_v1/receipts/builder_sandbox_canaries.json`:
    regenerated byte-identically; canonical SHA-256
    `3cfc64c0868c606cce9b400cf88493f2b9057d7c7dfe668cfd016b6114bf6238`;
    file SHA-256
    `ba1cb6246298d90f7011ed559c1d1829e636b0bf8bf62d82ae4d980146b5a4df`;
    unchanged profile SHA-256
    `f18216e5c08a0ac2eac8986f8338b032c84d2de24a231083ae28083bb061f44f`.
  - `experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`:
    canonical SHA-256
    `69a20c2da8185f6c435be3249f887e557b997255eef5c4a570e345ce0d3d6ce3`;
    file SHA-256
    `2f6276011998c6105ca6714f7b380575f4a1c49e8497aac471c0e35e55629ea6`;
    runtime fingerprint SHA-256
    `26f93064e481c3173315f59401dce0a839681c00da632cd981fb00b7b6a3e9fd`;
    source-tree SHA-256
    `e39d0b08c09813b987c15de0cd8e187a5e2f68c7e9ad964b9c27c00768c39e51`;
    worker-source SHA-256
    `3ab6818d75e5f2d015e9006be9a5a8dc96ff9c43ab823b9f6e0dc496a49ddac9`;
    unchanged eight-step smoke SHA-256
    `048cf64ca8b06f6bd9f00c1b591ba1d0b40fb87620114db6f3a3c367f4218bdc`.
  - `experiments/family_b_target_speed_v1/receipts/host_sandbox_canaries.json`
    remains the pre-B0FIX4 `18/20` profile-v3 diagnostic: canonical SHA-256
    `097006bf8becd4a6b4bed5ffb4dbe56174516e9a878a76a71827ac3d34bf7a97`;
    file SHA-256
    `52bbc11b50513fe3d83bb5c792ebbdf50c35b1069d960a418a8c9b4bcf67780e`.
  - `docs/operations/CURRENT_RESEARCH_HANDOFF.md`.
  - Verification: sandbox tests `24 passed, 9 skipped` with exact reason
    `sandbox-exec: sandbox_apply: Operation not permitted`; runtime receipt
    replay `8 passed`; full suite `44 failed, 1145 passed, 11 skipped`, with
    observed failure IDs exactly equal to the saved 44-test baseline; Ruff
    lint and format checks pass. No training or behavioral evaluation ran.
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

Fable resume: review the 03A2FIX scope-only diff and both regenerated schema-v2
receipts; refresh and review the B0 runtime receipt under reward-slice authority
because its repository-wide source-tree hash changed; rerun the full suite and
commit only when the non-baseline failure is gone. Do not run training or
credit import equivalence as behavior.

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

## B0 host verification under profile v3 `2026-09-04T21:58Z`

Host canaries: `18` passed, `2` failed (`network`, `tracing`, exit `4`);
`memory_abuse` passed with mechanism `timeout_kill`. Host receipt canonical
SHA-256 `097006bf8becd4a6b4bed5ffb4dbe56174516e9a878a76a71827ac3d34bf7a97`.
The two failures are probe defects: socket creation and `getpriority` are not
operations Seatbelt governs. Packet `TASK-20260904-B0FIX4` replaces them with
`connect`/`bind` and `libproc` probes. No profile change is needed.

## Two-lane orchestration `2026-09-04T22:15Z`

Samuel authorized concurrent Fable and Astra orchestration. ADR 0007 records
the lanes. Mailbox: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.git/harness-coordination`
through the Astra checkout's `scripts/research-mailbox` (`inbox fable`,
`ack fable ID`, `send --from fable --to astra ...`, `status fable`, `board`);
Fable writes to it under its lease with that directory in the audit scope.
Acceptance `20260904T221414.221774Z-a293af4af0eb4362a66d1a71554d3c15` replied
to proposal `20260904T201259.759134Z-ee5e9a7d67154a118dac6ffffd73e4c1`. Poll at
turn start, before dispatch or integration, after each worker, and about every
30 minutes. B0 ownership transfers only by a `handoff` message after its two
reviews. Heavy jobs over ten minutes need a mailbox proposal and a peer reply.

Current B0 host record (supersedes every earlier canary statement in this
file): profile v3, `20/20` canaries passed, `memory_abuse` mechanism
`timeout_kill`, host receipt canonical SHA-256
`5465426251311c147843488f8b92eb7b0fca0428fee4091267f41dde53adfdd9`, sandbox
tests `33 passed`, full suite `1200 passed`. Software-boundary evidence only:
B0 ran no training, learned-policy rollout, protected endpoint measurement,
or humanoid behavioral evaluation. Earlier sections describing `14 failed`,
`18/20`, or receipt `097006bf…` are superseded history.

## B0 reviews and handoff `2026-09-05T01:03Z`

Scientific review `09` (run `20260904T221620Z-cd03d6f4-3479-4c26-9410-ecd4c2d57dcd`) and robustness review `10`
(run `20260904T221621Z-d07a5b95-5955-462f-834e-61f108bda886`) of `89d4c34`: both `ACCEPT-WITH-REPAIRS`, no P0,
`6` and `10` P1. Shared findings: candidate code executes in the parent during
validation and calibration; admission and result lineage unbound; canary
ledger contradiction (the protocol still cites the pre-repair `18/20` receipt);
canaries that count any exception as containment. Scope note: the B0 commit
also carried ADR 0007 and the decisions index; recorded as an explicit
exception. A `handoff` message to Astra transferred the reward paths with all
findings and the condition that nothing executes or trains on B0 until the P1
repairs land and are re-reviewed. Fable remains promotion owner for `main`.

Importer repair `03A2FIX` (run `20260904T221621Z-0d2563da-2794-4f84-a220-bb7fd4525ab5`, thread
`01a06e7e-89f4-7fd2-b4cd-e9f93bde1e8d`): all ten findings implemented,
`173 passed, 1 skipped` focused; import receipt schema v2
`c2379a21079c40928548b593cb86a8fd8eea468787cf31c28481792d3121e80b`,
equivalence receipt v2 `0d5ec467051e2eebc16a0a186afbe85cc1364afbeab8fcbefa41f088cb656f31`;
NPZ and actor fingerprint unchanged. One full-suite failure outside the
baseline: the reward runtime receipt's whole-tree `source_tree_sha256`
(recorded `e39d0b08…`, current `7496476d…`), regenerated by Fable once and
recorded as a reward-lane P1.

Samuel's alignment question (via Astra, `20260904T221845…`) answered with an
independent interpretation of reference composition; recorded in the strategy
as the frame for Experiment 003: a three-gait composition task with
state-triggered transitions and recovery, LLM-designed oracle program,
feedback-driven revision, elapsed-time and hand-written baselines.

## Provenance correction for `425d3bc` and integration rule `2026-09-05T03:13Z`

Commit `425d3bc` bundled two integration-owned changes with the builder's
packet slice: the strategy addition (Experiment 003 composition frame) and the
regenerated reward runtime receipt
`experiments/family_b_target_speed_v1/receipts/builder_runtime_no_learning_smoke.json`.
Both were authored by Fable under its lease, outside packet `03A2FIX`'s
allowlist; the packet's changed-files ledger and the earlier "scope-only"
wording were therefore inaccurate. Reviews `11` and `12` flagged this
(`SCI-03A2FIX-01`, `R-07`). Rule from here on: builder slices and Fable's
integration edits are committed separately, and every integration edit is
listed in the handoff with its authority. Host verification results are now
recorded durably in `experiments/bootstrap_tqc_humanoid/reviews/fable_host_verification.json`.

## Importer hardening synthesis `2026-09-05T03:13Z`

Robustness review `12` of `425d3bc` (run
`20260905T010724Z-6ed23ba1-3467-46d6-8d8b-ce4cf60933be`): `R-01` rights set from
every registered payload; `R-02` release-bundle scanning; `R-03` bounds before
allocation; `R-04` parser exception normalization; `R-05` installed-file
identity; `R-06` atomic publication; `R-07` provenance (fixed above); `R-08`
boundary-level negative tests; P2 `R-09`, `R-10`. Fable accepts `R-01`,
`R-03`, `R-04`, `R-06`, `R-08` in full, narrows `R-02` to a git-index
guarantee until a release pipeline exists, binds locked artifact hashes and the
Python ABI for `R-05`, and closes the loop with one scoped closure review.
Packet `TASK-20260905-03A2FIX2` carries this; it runs after `03B`.
