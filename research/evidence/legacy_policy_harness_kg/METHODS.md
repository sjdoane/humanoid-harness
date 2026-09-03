# Corpus and extraction method

**Evidence cutoff:** 2026-08-31  
**Unit of review:** one public paper per independent research agent

## Question before search

> Which public results can change a decision about reference/oracle composition,
> executable task-reward design, human steering of those two artifacts, or the
> evaluation protocol—while the low-level controller and training system remain
> fixed?

## Search lanes

| Lane | Query families | Admission signal |
| --- | --- | --- |
| Oracle and modes | oracle-guided policy, multimode biped, preference-ranked modes, recovery transitions | produces or selects closed-loop references, guards, modes, or transitions |
| Whole-body trackers | SONIC, motion tracking, contact prompting, recovery, long-tail experts, hierarchical motion interfaces | defines the fixed controller contract, reachable reference set, or failure boundary |
| Reward design | LLM reward code, reward evolution, progress functions, language-to-reward, sim-to-real reward | generates, repairs, ranks, or validates executable reward programs |
| Steering | comparative language feedback, language-conditioned motion, inference-time policy steering | maps human intent to a permitted harness artifact |
| Evaluation | humanoid benchmarks, simulator suites, statistically reliable RL evaluation | changes the evidence required to claim improvement |

Searches used arXiv metadata and full text, exact-author lookup for Lokesh
Krishna, citation/project/code links, publisher records, and current web search.
The August 2026 sweep explicitly checked new `2608.*`, `2607.*`, and `2606.*`
humanoid motion-tracking records rather than relying on citation counts, which
are not meaningful for papers only days old.

## Selection policy

- `anchor`: directly defines the proposed controller/oracle/reward interface.
- `core`: provides a transferable mechanism, boundary, or ablation.
- `supporting`: useful adjacent evidence whose embodiment or optimization loop differs.
- `baseline`: supplies an evaluation environment or statistical protocol.
- Exclude papers that only share vocabulary, lack a public primary source, or
  require changing the fixed controller without informing an interface boundary.
- VIBE remains private meeting context. It is not represented as a public paper
  and no private diagram detail is attributed to SONIC.

## Independent extraction

Each admitted paper receives a separate agent and exactly two source-controlled
artifacts:

1. `cards/<arxiv-id>.md`: compact human-readable brief.
2. `extractions/<arxiv-id>.json`: queryable entities, parameters, relations,
   failure modes, evaluations, and a source-located evidence ledger.

Paper-local evidence labels are converted to globally unique graph keys as
`<paper-id>:<evidence-id>` during the deterministic build.

## Quality gates

- Primary paper first; official code/project and publisher only as supplements.
- Exact source version and URL recorded.
- Bare arXiv IDs, public source URLs, and unique paper-local evidence IDs are
  machine-checked.
- Every mechanism, parameter, failure mode, evaluation, and relation references
  at least one existing evidence item.
- Claims, author interpretations, and harness implications stay separate.
- A reported paper result never becomes a claim that RL-Sculptor implements it.
- Duplicate paper/entity IDs fail validation.
- Corpus state, one-agent-per-paper assignment, and card/extraction parity are
  audited before every build.
- Withdrawn or retracted work is quarantined and cannot produce an `INFORMS`
  edge to a project knob.
- Unknown relation endpoints become explicit concept nodes, never dangling edges.
- SQLite foreign keys and a full-text index make provenance and retrieval testable.

## Rebuild and query

```bash
python3 scripts/validate_extractions.py
python3 scripts/build_graph.py
python3 scripts/query_graph.py papers --tag reference_composition
python3 scripts/query_graph.py mechanisms --knob task_reward
python3 scripts/query_graph.py failures --text recovery
python3 scripts/query_graph.py evidence --paper 2403.04205
```
