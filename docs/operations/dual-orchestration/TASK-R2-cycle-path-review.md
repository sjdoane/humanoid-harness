# Review the minimum reward path for an executed cycle

- One read-only `gpt-5.6-sol/max` leaf; no delegation or candidate model calls.
- Work only in `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Baseline source: `754e869c618dbe4f1b43c4f1a513c023ec139306`.
- Parent added only coordination docs and an ignored active pointer after R1.
- Maximum 20 minutes; final at most 650 words. No edits, tests, workers, probes,
  simulator, training, installation, messages, or runtime reward execution.

## Read

- `AGENTS.md`, `docs/operations/dual-orchestration/README.md`, project charter.
- In that docs directory: `CYCLE_CONVERGENCE_PROPOSAL.md`, `R2_PLAN_RESULT.md`,
  `R1_ACCEPTANCE.md`, `A3_RESULT.md`, and `A1_RESULT.md`.
- Exact R2 plan:
  `.orchestration/sol-runs/20260905T171030Z-de04d66f-5dcd-4117-ad96-c64fb4f0e357/final.txt`.
- Relevant source: `rewards/contract.py`, `static_validation.py`, `sandbox.py`,
  `_sandbox_worker.py`, `scale_calibration.py` under `src/oracle_composition/`;
  existing `reward_search/` proposal/feedback interfaces and their tests.
- Read main's committed ADR 0008 with `git show a0a0a3b:docs/decisions/0008_alignment_pivot_and_tracker_family.md`
  if needed. Do not use Fable's dirty runner source as an agreed interface.

## Decision requested

Compare two **engineering routes**, without reopening the scientific goal:
(A) minimum R2/R3 isolated Python runtime; (B) a newly declared, data-only JSON
formula/parameter family evaluated by trusted code for the first exploratory
reward cycle. B is a smaller capability, not arbitrary reward code and not a
claim that the old sandbox findings vanished.

Choose the smallest defensible route from actual source, or reject both with
the exact missing interface. Give at most three blocking issues, minimum new
or changed files, a concrete first builder packet, required negative tests,
and what must wait for Fable's committed runner/report and resource agreement.
Preserve R1 refusal, evaluator independence, immutable input/feedback binding,
bounded input/compute, matched-factor baselines, and historical receipts.
The recommendation does not authorize host or generated-code execution.

Do not produce a second broad architecture or containment research agenda.
Return progress/bottleneck/next step, then the decision and bounded next slice.
