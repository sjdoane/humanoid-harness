# Literature-review protocol — question approved; remaining gates pending

**Run ID:** `001_oracle_composition`
**Type proposed:** systematic scoping review plus mechanism/evidence map; no
meta-analysis
**Cutoff proposed:** 2026-09-02
**Status:** guiding question approved; candidate discovery only; formal
screening has not started

## Guiding question

With Gymnasium MuJoCo Humanoid dynamics, observation/action contract, reference
tracker, tracking objective, task reward, trainer, budget, and evaluator held
fixed, which state- and phase-conditioned reference-oracle mechanisms improve
multi-skill transitions and perturbation recovery over fixed or open-loop
reference playback?

**Approval:** approved by the researcher in this task on 2026-09-02. This
approval covers only the wording above.

## Purpose, audience, and claim boundary

- **Purpose:** identify evidence-bounded oracle-mechanism hypotheses, failure
  modes, and benchmark requirements for later controlled experiments.
- **Audience:** the RL-Sculptor research team and future reviewers of the
  semester paper.
- **Boundary:** the review can justify what to implement and test. It cannot
  establish that any mechanism improves the frozen Gymnasium tracker; only the
  preregistered behavioral experiments can establish that result.

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

## Proposed screening fit classes

| `fit_class` | Operational meaning |
|---|---|
| `direct_fit` | Meets the primary-source and embodiment criteria and evaluates an explicit reference-composition, state/phase, switching, transition, feasibility, or recovery mechanism in physics-based control. |
| `transferable_method_fit` | Supplies an applicable mechanism or failure boundary, but its controller, embodiment, task, or evaluation prevents direct support for the frozen-tracker claim. |
| `background_only` | Supplies historical, benchmark, runtime, or methods context without mechanism evidence for the review question. |
| `exclude` | Meets at least one exclusion criterion; record the first decisive reason and any secondary reasons. |

These labels will be applied only after the exact source is registered and the
formal screening gate opens. The current A/B/C/H candidate queues are not
`fit_class` assignments.

## Search lanes

1. Reference motion + composition/sequencing/transition/motion graph.
2. Motion phase/phase manifold/local phase + humanoid or character control.
3. Oracle/scheduler/hybrid automaton/skill graph/motion matching.
4. Recovery/perturbation/reacquisition + humanoid tracking.
5. Retargeting/feasibility/controller awareness/cadence/root frame.
6. Gymnasium or MuJoCo Humanoid + reference/imitation/tracking/phase.
7. Skill chaining/policy switching/transition policy.

Search arXiv and OpenAlex from 2002 to the cutoff. ACM Digital Library, IEEE
Xplore, RSS, CoRL/PMLR, and OpenReview are primary-record resolution surfaces,
not independently searched Boolean indexes. Admit older foundations only
through backward chaining. The exact literal queries, filters, pagination,
exports, and citation route are proposed in
`FORMAL_SEARCH_QUERIES_DRAFT.md`. Record query, date, result count, version,
and every screening decision. The proposed stopping rule is two consecutive
complete backward/forward citation rounds with no new `direct_fit` primary
source after all database lanes have run.

## Approval gate

The guiding question is approved. The researcher must still approve or revise
the review type and outputs, search strategy and stopping rule,
literal query ledger, inclusion/exclusion criteria, and metadata-only candidate
follow-up queues before formal search or screening. Source triage and corpus
adequacy, the final included set, coding taxonomy, synthesis, and final report
each retain a later human gate.
