# PRAXIST engineering-method record

## Classification

| Item | Assessment |
|---|---|
| Source | Li et al., *From Experimental Artifacts to Solution Lineages*, arXiv:2608.25955v1 |
| Evidence lane | Engineering method and orchestration |
| Direct oracle evidence | No |
| Local exact PDF | Registered and Git-ignored |
| Formal review status | Outside oracle-mechanism screening until the review scope is approved |

## What PRAXIST does

PRAXIST turns a research campaign into a repeated, evaluator-grounded artifact
cycle:

```text
design contract -> reproducible artifact -> external evaluation -> typed finding
       ^                                                       |
       |                                                       v
 next agenda <- confirmed/candidate/diagnostic/validation frontier <- PI synthesis
                              |
                              +----> lineage + durable lessons
```

- Parallel peers receive deliberately different design contracts.
- Each contract states the mechanism, intervention surface, intended evidence,
  validation hook, parent lineage, and forbidden changes before construction.
- The task's external evaluator—not the agent's prose—scores artifacts.
- Results become typed findings and enter role-specific frontier lanes.
- Later generations inherit selected evidence, failures, open validation debt,
  and an agenda rather than raw transcripts alone.
- A lineage records which artifacts, findings, and decisions led to the final
  reported artifact.

## How it can help this project

After the tracker and protected evaluator are valid, PRAXIST can coordinate
bounded oracle variants, preserve useful failures, assign causal ablations, and
keep later candidates tied to the exact parent mechanisms and evaluator
receipts. That is directly useful for oracle generation and iterative refinement.

It cannot establish the missing substrate:

- It does not supply a Gym Humanoid reference-conditioned tracker.
- It does not decide what counts as a valid task metric.
- It cannot turn an appended command into causal reference use.
- Its paper does not show that a phase-aware humanoid oracle works.

For this repository, PRAXIST is therefore **more useful as the experiment and
lineage layer than as the oracle algorithm itself**.

## Paper-aligned cautions

- Configured compute budgets are ceilings, not measured use.
- Generations, peer sessions, optimizer updates, simulator steps, and episodes
  are distinct units and must be reported separately.
- Several paper case studies have one run or limited repetition; their process
  examples should not be read as universal performance guarantees.
- Task-owned evaluators and frozen intervention boundaries remain prerequisites.

