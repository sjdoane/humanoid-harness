# System lead: complete the learning loop

| status | current truth |
|---|---|
| progress | Samuel assigns Astra both implementation lanes; the role-change message is sent to Fable. |
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
- Communication is queued until Fable reads it; delivery is not acknowledgment.

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
- Alternative-MDP decision pending primary-source review: native Humanoid,
  GMT/G1, HumEnv/Meta Motivo, SONIC. Require accessible weights/data, license
  clarity, reference ABI, Mac feasibility, task-reward training, and lower total
  integration cost. A more elaborate environment is not itself progress.

## Procedure acknowledgment

Experimental-design and SB3 skill procedures informed the bounded comparisons
and runtime checks. Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner,
A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
Metadata checked 2026-09-06 (current arXiv record: v2); no task-level efficacy is
inferred from using these procedures.
