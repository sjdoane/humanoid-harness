# Literature-review protocol — draft pending human approval

**Run ID:** `001_phase_aware_humanoid_oracles`  
**Type proposed:** systematic scoping review plus mechanism/evidence map; no
meta-analysis  
**Cutoff proposed:** 2026-09-02  
**Status:** candidate discovery only; formal screening has not started

## Guiding question

With Gymnasium MuJoCo Humanoid dynamics, observation/action contract, reference
tracker, tracking objective, task reward, trainer, budget, and evaluator held
fixed, which state- and phase-conditioned reference-oracle mechanisms improve
multi-skill transitions and perturbation recovery over fixed or open-loop
reference playback?

## Proposed outputs

- Concise Markdown synthesis.
- Version-pinned source cards and extraction records.
- Mechanism-to-design-requirement table.
- Failure taxonomy and benchmark protocol.
- CSV screening, evidence, and citation ledgers.
- BibTeX generated from the included primary sources.

## Inclusion criteria

- Primary peer-reviewed paper or version-pinned arXiv paper available by the
  cutoff.
- Physics-based humanoid, biped, or articulated-character control.
- Explicit reference, phase, mode, skill graph, scheduler, transition,
  feasibility, or recovery mechanism.
- Adjacent animation work only when it directly informs phase estimation,
  motion matching, transition construction, or state-machine design.
- Evaluation-method papers only when they change the evidence required for an
  oracle claim.
- Official code and project pages may supplement a paper but never replace it.

## Exclusion criteria

- Reward-only work with no reference-composition implication.
- Motion generation with no transition, feasibility, controller-interface, or
  recovery relevance.
- Single-skill tracking work with no applicable interface or failure boundary.
- Surveys, blogs, leaderboards, and repository claims as scientific evidence.
- Withdrawn or retracted work; Athena-WBC remains quarantined.
- Sources whose exact bytes and version cannot be acquired and registered.
- Different-controller evidence as direct support for the frozen-tracker claim;
  retain it only as transferable mechanism evidence.

## Search lanes

1. Reference motion + composition/sequencing/transition/motion graph.
2. Motion phase/phase manifold/local phase + humanoid or character control.
3. Oracle/scheduler/hybrid automaton/skill graph/motion matching.
4. Recovery/perturbation/reacquisition + humanoid tracking.
5. Retargeting/feasibility/controller awareness/cadence/root frame.
6. Gymnasium or MuJoCo Humanoid + reference/imitation/tracking/phase.
7. Skill chaining/policy switching/transition policy.

Search arXiv, ACM Digital Library, IEEE Xplore, RSS, CoRL/PMLR, and OpenReview
from 2002 to the cutoff. Admit older foundations through backward chaining.
Record query, date, result count, version, and every screening decision. The
proposed stopping rule is two consecutive backward/forward citation rounds with
no new direct-fit source after all search lanes have run.

## Approval gate

The researcher must approve or revise the guiding question, review type,
inclusion/exclusion criteria, and output package before formal screening.

