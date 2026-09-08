# System lead: complete the learning loop

Historical plan. The [2026-09-08 plan](POST_TRAINING_PLAN_20260908.md) supersedes
its priorities and roles. Preserve the earlier stop rules and evidence; do not
restart native tracker work or wait for Fable from the instructions below.

| status | current truth |
|---|---|
| progress | Fable acknowledges Astra as system lead; diagnosis, development ablation, and numeric-only GMT admission are integrated. |
| bottleneck | No trained reference-composition result; diagnosis, proposal, training, and revision are not joined end to end. |
| next step | Finish the clean peer handoff, run a bounded native-tracker utility test, and implement evidence-to-revision routing. |

## Authority and completion

- Authority: Samuel's direct 2026-09-06 request; supersedes the equal lane split.
- Astra: architecture, implementation, integration, bounded training, evidence.
- Fable: independent scientific/code reviews, criticism, and joint ideation.
- Requested worker configuration: Sol/max, bounded independent tasks, no nested
  delegation. Requested settings are not served-model attestation.
- User authority includes local training toward this objective. It does not
  include paid cloud resources, publication, private-source disclosure, or
  bypassing provider safeguards.
- Mailbox handoff: `20260906T164045.084189Z-f91211dc986c4788a2d16ac759c458c9`.
- Fable acknowledgment: `20260906T175818.868830Z-6266274cc43349c9a49af2daa9e7be95`.
  Astra also owns main promotion after Fable's clean T2C2 checkpoint.

Completion requires all of these, not just a runnable command:

1. Supplied reference clips reach one reference-conditioned policy numerically.
2. Observable state selects meaningful transitions; playback-only is a control.
3. The task reward affects training separately from the fixed tracking reward.
4. Real rollout evidence informs an LLM proposal and at least one revision.
5. Matched evaluation retains failures and shows the outcomes of both artifacts.
6. A replay displays task state, reference selection, speed/error, and transitions.
7. Exact inputs, budgets, generated artifacts, decisions, and results are retained.

A one-seed demonstration is exploratory. Robustness and cross-MDP performance
require separate studies; no claim of a foolproof or universally necessary method.

## Architecture

```text
task + supplied references + initial reward + optional correction
                              |
                         design agent <---- relevant literature/case evidence
                              |                         ^
                          oracle O, reward r            |
                              v                         |
                         validation                     |
                              v                         |
 fixed adapter: state -> O -> reference window -> policy training
                tracking reward (fixed) + task reward r  |
                              v                         |
                 policy + synchronized rollout          |
                              v                         |
                independent evaluator -> diagnosis -----+
```

- LLM revision runs between experiments; deterministic oracle execution runs
  at control cadence. Do not insert a remote model into each motor step.
- Separate authorable artifacts from the frozen trainer, reference format,
  controller initialization, tracking reward, evaluator, and budget.
- The native adapter learns reference use and later fine-tunes the actor. It is
  not an already-trained frozen reference tracker. Freeze the same initialization
  and recipe across arms; label a new adapter as a separate experiment family.
- Reuse existing validators, persistence, graph queries, and runtime adapters.
  Do not add a second graph database or a parallel experiment authority.

## Implementation priorities

| order | deliverable | stop/review rule |
|---|---|---|
| 1 | Clean peer handoff; essential spawn/lineage repair only | Do not duplicate active peer work |
| 2 | Existing native T1 utility smoke | One 1,200 s / 196,608-transition attempt; retain failure and reassess route |
| 3 | Compact phase/subtask diagnosis, missing-evidence routing, grounded revision | Unit fixtures are plumbing evidence only; consume real output before a behavior claim |
| 4 | Baseline and candidate reward training with fixed oracle | Same initialization, trainer, seeds, budget, evaluator, checkpoint rule |
| 5 | Oracle change with fixed reward; one feedback-informed revision | Exact/zero/shuffled/time-shifted reference controls before oracle-quality claims |
| 6 | Combined task and labeled replay | Lock O1 and r1 before four-arm comparison; joint re-search is a separate algorithm |

- Native utility failure is not permission to keep rebuilding a tracker
  indefinitely. Compare a supplied tracker before a second native training design.
- No new general-purpose hardening layer unless it blocks execution, changes a
  scientific conclusion, or fixes a concrete correctness/security defect.
- One heavy job at a time; initially at most 4 compute threads and 12 GiB RSS,
  with tighter existing per-study limits preserved. Record actual wall/CPU/RSS.
- No silent retraining, seed shopping, expanding a run cap, or held-out reuse.
- Report insufficient power honestly; more episodes from one policy are not
  independent training seeds.

## Failure-to-action policy

| evidence | next action |
|---|---|
| No verified reference installation/use | Adapter check; no oracle-quality claim |
| Transition-local failure with competent single clips | Examine boundary/phase; propose an oracle patch |
| Poor tracking throughout a single clip | Adapter/controller diagnosis; do not blame reward alone |
| Good tracking, wrong speed/task completion | Propose task-reward revision with the oracle fixed |
| Missing signal or ambiguous cause | Request a specific measurement or one targeted human question |
| Baseline already satisfies the task | Stop or declare a harder separate task; do not manufacture improvement |

## Reuse and research rationale

- RL-Sculptor `diagnose.py`, `oracle_gen.py`, and the iteration loop: reuse
  component evidence, seed dispersion, reversions, human correction, and replay
  screens as small modules; keep the old workspace read-only.
- [OGMP v3](https://arxiv.org/html/2403.04205v3): state-conditioned finite-horizon
  references; current switch-only phase matching is a restricted baseline.
- [Eureka v2](https://arxiv.org/html/2310.12931v2): measured component reflection
  and independent fitness between candidates; avoid transferring its compute
  budget blindly.
- [RDA v1](https://arxiv.org/html/2606.01672v1): subtask failure and instruction
  alignment inform revision. Use vision when it adds missing information, not
  as a replacement for available simulator measurements or protected scoring.
- MDP decision: one existing native smoke, not a commitment to native tracking.
  GMT/G1 is the leading alternative because it supplies demonstrated reference
  competence and Mac deployment code. Admission and our adaptation remain untested.
  Missing original PPO code does not rule out a frozen base plus a trainable
  higher-level policy. No downloaded TorchScript or pickle has been executed.
- A new high-level-policy adapter is a new experiment family: freeze its state,
  action, cadence, tracker, training recipe, and objective metrics. Do not let an
  oracle candidate change the action interface or introduce a residual halfway
  through a comparison. Preserve an independently trainable policy downstream
  of the two authored artifacts; do not rename oracle scripting as policy training.
- Evaluate GMT first on supplied motions and numeric-reference interventions,
  then one meaningful splice, then reward authority. Stop a route that fails
  competence or safe artifact admission; count integration time as well as compute.
- Fable's follow-up review supports native T1 first, then the fixed-oracle T2
  reward loop. That is useful reward evidence, not the complete two-knob result.
  Its under-five-hours estimate is unmeasured. A protected metric implementation
  may score development data; untouched held-out outcomes must never select a
  revision. Do not wait weeks for a possible tracker without an availability date.
- HumEnv/Meta Motivo remains a legitimate separate latent-composition hypothesis,
  not a drop-in PPO result. SONIC currently adds a Linux/NVIDIA training dependency.
  See `MDP_AND_FABLE_REVIEW_20260906.md` for evidence and review limits.

## Procedure acknowledgment

Experimental-design and SB3 skill procedures informed the bounded comparisons
and runtime checks. Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner,
A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
Metadata checked 2026-09-06 (current arXiv record: v2); no task-level efficacy is
inferred from using these procedures.
