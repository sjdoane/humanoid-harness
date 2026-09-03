# Knowledge-graph ontology

## Node types

| Type | ID form | Meaning |
| --- | --- | --- |
| `project_component` | `system:*`, `artifact:*`, `input:*`, `boundary:*` | Historical project loop and its contract |
| `project_knob` | `knob:*` | The two allowed artifacts plus steering/evaluation/fixed-boundary classification |
| `paper` | arXiv ID, for example `2403.04205` | One independently reviewed public paper |
| `mechanism` | paper-specific stable ID | Reported method component or causal proposal |
| `parameter` | generated `<paper>:parameter:<n>:<name>` | Exact reported value with units, context, and evidence |
| `failure_mode` | paper-specific stable ID | Reported or source-supported limitation/confound |
| `evaluation` | paper-specific stable ID | Dataset, benchmark, metric, ablation, or protocol |
| `concept` | namespaced semantic ID | Explicit endpoint used by one or more paper relations |

## Edge provenance

| Provenance prefix | Authority | Evidence rule |
| --- | --- | --- |
| `paper_extraction:<paper-id>` | Public primary source | Must reference a paper-local evidence item; graph build globalizes it as `<paper-id>:<evidence-id>` |
| `curatorial:decision_relevance.knobs` | Corpus review decision | Classifies a paper; it is not an author claim |
| `sanitized_context:legacy_project_contract` | Sanitized historical project context | Defines historical project edges; never attributed to a public paper |

## Project traversal

```text
reference_dataset ─┐
task_reward_0 ─────┼─INPUT_TO─> sam_harness
human_steering ────┘               │
                    ┌───────────────┴───────────────┐
                 PRODUCES                        PRODUCES
                    │                               │
                 oracle_k                     task_reward_k
                    └──────────CONFIGURES───────────┘
                                    │
                           parameterized_mdp
                                    │
                         policy → evaluator → metrics
                                    └──FEEDBACK_TO──> sam_harness
```

## Retrieval invariants

- Project context, curator classification, and paper claims are always separable.
- Paper-local evidence labels may repeat; generated graph keys may not.
- Unknown relation endpoints become named concept nodes rather than dangling strings.
- Parameters retain their paper wording; cross-paper normalization happens through
  full-text search and project-knob edges, not by erasing units or context.
- A mechanism can inform a knob without being admissible under the frozen trainer.
  The paper card and `not_evidence_for` field carry that boundary.
- No public `VIBE` paper node exists until a citable preprint is supplied.
