# First-principles review: composition, scale, and adaptation

| status | current truth |
|---|---|
| progress | Targeted primary-source review supports a narrower, testable motivation for the two-knob harness. |
| bottleneck | No matched result establishes that our harness beats prompting, scripted composition, or a capable pretrained tracker. |
| next step | Use the staged references for a capped real-reference mechanism test; compare simpler alternatives before joint adaptation. |

- Date: 2026-09-06. Owner: Astra, oracle lane.
- Status: advisory decision review; proposed cross-lane changes await Fable.
- Scope: Samuel's request to revisit first principles and find a feasible joint demo.
- Not the completed formal literature review. Its approval log and inclusion tables remain unchanged.
- Source locators and access depth: [evidence ledger](FIRST_PRINCIPLES_SOURCES_20260906.md).
- Private meeting/proposal material informed interpretation; no raw transcript is reproduced here.

## Decision

**Drop the universal necessity claim. Keep the two-knob experiment.**

Proposed research question:

> Given an available pretrained controller, a fixed MDP and reference library,
> and a bounded adaptation budget, when does explicit reference-oracle or
> task-reward revision improve held-out task success and recovery beyond
> fixed-reference, scripted, and prompt-conditioned alternatives?

- Scale and in-context learning may solve tasks of interest. We have no evidence that they cannot work for humanoids.
- A fixed reference is not an open-loop action sequence. A reactive tracking policy already closes a feedback loop.
- The contribution must be a measured adaptation benefit, not the existence of a VLM, knowledge graph, or orchestration loop.
- A negative result is useful: if prompting or a handwritten composer suffices, do not spend training budget to reproduce it.

## Four different loops

| Component | What stays fixed or changes | Feedback location |
|---|---|---|
| Reference data | Supplied motion samples may remain fixed | None required inside the stored data |
| Tracking policy | Current state plus reference produces actions | Robot state every control step |
| Oracle/composer | Selects, aligns, transitions, or rejoins references | Observable task/robot state and local phase |
| Design harness | Proposes a new oracle or task reward, then evaluates learning | Episode evidence across experiments |

GEN-1.5 places demonstrations beside rolling observations; its executor is not
blind replay. Learned policies can internalize selection, transition, and
recovery. Explicit oracles are another way to represent those operations, not
their only possible implementation. [GEN-1.5](https://generalistai.com/blog/gen-1.5)

```text
INPUT: task + compatible reference library + current robot/task observations
                     |
          cheapest eligible adaptation route
          /                  |                    \
   prompt/demo         oracle revision       task-reward revision
  (if supported)       O_k: state -> HxD       r_k: state -> reward
          \                  |                    /
           FROZEN adapter, controller family, tracking reward,
           scene, observations/actions, dynamics, trainer/budget
                              |
                  OUTPUT: policy rollout + costs
                              |
                 FROZEN independent evaluator
                              |
             FEEDBACK: success, failures, uncertainty
                  -> retain / diagnose / revise / stop
```

External systems with different controllers or embodiments belong in a separate
system-level benchmark, not in the frozen-family causal comparison above.

## What the counterexamples change

| Evidence | Implication | Limit of the inference |
|---|---|---|
| GEN-1.5 reports one-shot manipulation and qualitative composition of demonstrations | Treat prompted composition as a serious alternative | Company report; not a matched humanoid comparison. Reported fine-tuning results are not ICL results |
| ICRT and Instant Policy execute new manipulation tasks from demonstration context | In-context adaptation is more than a speculative future baseline | Their available interfaces are not Humanoid-v5 controllers |
| RoboTTT improves with longer context and uses test-time fast-weight updates | Context and test-time adaptation deserve explicit cost accounting | Not ordinary frozen-weight ICL; whole-body transfer remains a separate question |
| SONIC scales humanoid tracking and adds planner/VLA interfaces | Reuse strong trackers; do not rebuild a foundation controller as the research contribution | A new G1/SONIC experiment changes the adapter, not just the oracle |
| MCP and OGMP already study composable skills/oracle-guided modes | Composition and closed-loop oracles are established mechanisms | Automating their design may still help; novelty and benefit require comparison |

Sources: [ICRT](https://arxiv.org/abs/2408.15980v2),
[Instant Policy](https://arxiv.org/abs/2411.12633v2),
[RoboTTT](https://arxiv.org/abs/2607.15275v1),
[SONIC](https://arxiv.org/abs/2511.07820v4),
[MCP](https://arxiv.org/abs/1905.09808v1),
[OGMP](https://arxiv.org/abs/2403.04205v3).

## Why a harness, rather than just ICL?

This is not an either/or choice.

- **In-context design iteration:** the LLM reads failures and revises oracle/reward artifacts in context. The downstream robot policy may still require RL. Eureka already uses this pattern; do not label it embodied policy ICL.
- **Policy ICL:** a compatible pretrained sensorimotor policy consumes demonstrations and current observations without task-specific weight training.
- **Reward-model prompting:** demonstration-conditioned prompt optimization can improve reward estimates before policy learning. Demo2Reward is a relevant adjacent approach, not evidence that downstream RL disappears.

The harness is motivated when the available controller lacks a useful task
prompt interface, when task constraints are not met by existing behavior, or
when inspectable state-dependent references/rewards offer cheaper adaptation.
Those are conditional engineering advantages to measure—not a proof against
end-to-end scale. [Eureka](https://arxiv.org/html/2310.12931v2),
[Demo2Reward](https://arxiv.org/abs/2606.00083v1)

Do not describe this study as unrestricted **MDP tuning**. The authorable
objects are `O` and task `r`; the other MDP/controller/training components stay
fixed within a study. DrEureka's domain-randomization search is adjacent work,
not permission to mutate our frozen dynamics. [DrEureka](https://arxiv.org/html/2406.01967v1)

## Does vision earn its cost?

- Use structured state/contact/phase metrics first when they identify the failure.
- Add video when it supplies missing semantic or behavioral evidence.
- Compare structured-feedback LLM versus the same pipeline plus visual feedback.
- A visual judge is not ground truth. Calibrate on success, failure, near-miss, and misleading-view examples; allow abstention.
- Keep final task success and protected guardrails outside generated reward code and outside the candidate-selecting judge.

RDA already revises reward code from visual rollout diagnosis, including four
HumanoidBench tasks. Its final alignment metric uses another VLM, while task
success uses benchmark verifiers. This supports visual diagnosis as a useful
candidate mechanism, not universal visual necessity or human-verified alignment.
Its authors also report costly RL search and conflicts between early and late
subtask rewards. Phase-aware oracle/reward interaction is therefore a concrete
question worth testing, not an established solution. [RDA, Sections 5–6](https://arxiv.org/html/2606.01672v1)

RoboReward documents uneven VLM reward accuracy and the importance of negative
and near-miss data. Use its lesson to design validation cases; do not assume a
larger general VLM is automatically the better evaluator. [RoboReward](https://arxiv.org/html/2601.00675v2)

## Minimal comparison that could reject our idea

Two distinct comparison layers:

1. **Mechanism attribution:** one controller/adapter family, locked candidates,
   matched training-seed blocks and training budget, independent evaluator.
2. **Practical alternative:** best accessible prompt-conditioned/scaled system,
   explicitly disclosing different embodiment, pretraining, data and hardware.

| Frozen-family arm | Oracle | Task reward | Interpretation |
|---|---|---|---|
| A | `O0` fixed reference schedule | `r0` baseline | Reactive tracking baseline; not open-loop actions |
| B | `O1` revised composition | `r0` | Oracle effect |
| C | `O0` | `r1` revised task reward | Reward effect |
| D | `O1` | `r1` | Combined effect and interaction |

- B and D must share the exact locked `O1`; C and D must share the exact locked `r1`. No extra D-specific search or training budget. Coupled co-adaptation is a separate algorithm comparison, not this factorial interaction cell.
- Add a handwritten state-aware composer as a strong simplicity baseline.
- Genuine policy ICL requires a compatible model and demonstration interface. Do not relabel JSON generation or controller switching as policy ICL.
- Primary outcome: independently measured held-out task success; report falls, transition/recovery failures and tracking error separately.
- Adaptation costs: simulator steps, demonstrations, tokens/calls, wall time, hardware, human edits and inference latency. Report pretrained-model cost separately.
- Train/development/test splits must separate source trajectories or tasks, not merely neighboring frames. The independent experimental unit is a trained seed, not an episode or video frame.
- Retain failed trials. Freeze stopping and checkpoint rules before running; choose cohort size from pilot variability and compute constraints, not an invented power claim.
- Exact/zero/shuffled/time-shifted numeric-reference ablations must establish causal use before claiming oracle effectiveness.
- **Mechanism test:** estimate B/C/D effects against A and the handwritten composer within the matched family. Failure to show a useful gain leaves benefit unestablished; reference insensitivity invalidates the oracle-performance claim.
- **Deployment choice:** compare an external system only on genuinely matched task goals/cost accounting; different embodiment or controller prevents causal oracle/reward attribution. Prefer the simpler eligible system when it meets a preregistered practical/non-inferiority margin at lower cost.
- Define that margin before evaluation. A nonsignificant difference in a small pilot is not evidence of equivalence.

This design does not authorize a cohort or alter the frozen T1/T2 studies.

## Route to an honest visual result

First use the existing T1 fast/slow/fast schedule as a **mechanism smoke**, not
as evidence of state-aware task composition. Its numeric references are now
locally staged; [setup receipt](REFERENCE_LIBRARY_SETUP_20260906.md).

- Proposed cap: one existing non-promotable smoke, at most 196,608 transitions
  and 45 minutes, whichever comes first; exact resource approval still required.
- First establish worker installation/consumption of numeric references. A
  later causal-use comparison must show they affect behavior; input shape alone
  is not sufficient.
- No automatic retries or larger training budget. A timeout is an incomplete
  test, not evidence that composition is impossible.
- If this bounded route cannot establish the mechanism, record the local
  adapter gap and assess a supplied lab tracker or a separate existing-tracker
  pilot. Do not turn this project into open-ended foundation-tracker training.

The **locomotion-only speed-zone course is deferred** until those gates pass:

- Desired behavior: move along a straight course, slow in a marked zone, then resume the faster gait.
- Oracle role: choose compatible gait segments, align their reference frames, and transition/rejoin using observed progress rather than elapsed time alone.
- Reward role: adapt the tracking policy toward the requested speed/zone objective without falling or exploiting the scorer.
- Display: task zones, actual/target speed, active reference, phase, transition reason and failures synchronized with the rollout.
- Perturbation test: delay progress or vary start placement; a time-only switch and a state-aware switch should react differently.
- Evidence: the same controller family must consume actual numeric reference windows and the authored task reward; no hidden switching between complete pretrained controllers.

This is a **new proposed task**, not a configuration claimed runnable today.
First establish a single-reference tracking baseline. Advance to two admitted
gaits only if their transitions are trackable. If the controller cannot support
it within the bounded pilot, investigate an existing compatible tracker rather
than expanding custom tracker engineering indefinitely.

Do not change observation schemas, reward authority, task zones or perturbation
distributions inside an already frozen comparison. Version the new task and
obtain cross-lane design/resource agreement first.

## Immediate actions and stop conditions

- Completed: staged exact native reference bytes and ten bound controller/receipt artifacts locally; verified library byte bindings and decoded only the 36 training-split clips. No simulator ran.
- Preserve the DeepMimic source-format audit: three clips pass that narrow audit; face-down get-up remains rejected. None is admitted as a Humanoid-v5 reference by that result.
- Completed: the bounded manifest-admission slice passed independent static review and 20 parent-reproduced pure tests. Other runtime/calibration gates remain; this is not research success.
- Ask Fable to independently assess this claim correction and propose a minimal joint task, with exact blocked interfaces and a capped pilot.
- No new heavy job, large model download, paid service, blanket environment migration or training cohort is implied by this note.
- Next evidence checkpoint must be a runnable reference-consumption test or an explicit adapter/no-go finding, not another generic orchestration feature.

## Independent challenge review

- A separate Sol/max reviewer checked the counterargument and this synthesis.
- Incorporated: clear separation of policy ICL from design iteration; exact
  shared factorial candidates; separate causal and deployment decisions;
  no equivalence claim from a nonsignificant pilot; capped task escalation.
- Research benefit remains a hypothesis. Cross-lane strategy promotion still
  requires Fable's explicit assessment; no reply is inferred from delivery.
