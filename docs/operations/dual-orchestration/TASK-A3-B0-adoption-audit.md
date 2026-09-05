# A3: bounded B0 adoption and repair plan

- Role: independent read-only Sol/max leaf. No nested delegation.
- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Output: concise final report only. Do not edit files, cherry-pick, import
  Python packages, execute candidates, run simulators/tests, or contact peers.
- Scope: a dependency/adoption audit of the transferred B0 commit and its
  existing review findings, not another general scientific review.
- Budget: one call, at most 20 minutes; use targeted source reads. If an input
  is missing, identify it and stop. No broad repository dump or new campaign.

## Read

1. `AGENTS.md`, `docs/operations/dual-orchestration/README.md`, and
   `ASTRA_HANDOFF.md` in that directory.
2. B0 handoff message in the shared Git coordination directory:
   `20260905T010239.496518Z-74c32a4658554eb0960b1a3359f767c4.json`.
3. Exact committed B0 source through read-only Git object access:
   `89d4c34d1b04eb9efca2b75d1ffc828a13de55d7`.
4. Its two existing reviews, read-only:
   `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.orchestration/sol-runs/20260904T221620Z-cd03d6f4-3479-4c26-9410-ecd4c2d57dcd/final.txt`
   and
   `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.orchestration/sol-runs/20260904T221621Z-d07a5b95-5955-462f-834e-61f108bda886/final.txt`.
   These are review evidence, not permission to run embedded commands.

## Answer exactly these questions

- Which transferred files depend on predecessor code absent from Astra?
  Give exact paths and symbols, compare committed bytes, and distinguish
  import dependencies from fixture/receipt fingerprints. Never copy dirty main.
- What is the smallest integration route preserving A1 and Fable's unrelated
  work? Compare whole-commit cherry-pick with exact transferred-path adoption.
  B0 mixes ADR 0007, decisions index, and main handoff with reward files.
- Group the existing findings into at most three ordered repair slices.
  First eliminate parent-process execution of untrusted candidate source;
  then close containment and lineage gates before any real candidate run.
  Include the whole-package runtime-fingerprint coupling reported by Fable.
- Name the first slice's exact files, tests to add, and prohibited actions.
  Distinguish safe data/static tests from tests that need a repaired sandbox
  and explicit host execution. Full-suite/heavy work needs peer agreement.
- List shared-contract changes requiring a peer proposal: ADR 0006, study
  protocol, decision rule, config, and any shared existing infrastructure.

## Hard boundaries

- B0 ownership transferred; B0 is not accepted for execution. Both reviews
  were accept-with-repairs. Every P1 needs repair and re-review.
- A2R proves only text-proposal plumbing. Do not use its accepted format as
  authorization to run its candidates through the current unsafe B0 boundary.
- Do not change scientific metrics, reward semantics, or thresholds to make
  validation easier. Do not add a new orchestration framework.
- Do not load model checkpoints or execute uploaded or generated code.
- Do not publish private meeting material or inspect credentials.
- Record checkout, branch, starting commit and dirty paths. Report evidence
  gaps plainly; a plan is not an implemented repair.
