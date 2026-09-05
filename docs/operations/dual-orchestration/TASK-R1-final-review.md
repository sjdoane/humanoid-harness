# R1 targeted closure review

- One read-only Sol/max leaf; no delegation, edits, peer messages, or model calls.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- At most 20 minutes; return a verdict or partial findings before the deadline.
- Baseline HEAD: `67722402c58374cf7f22347f7055abb979c9436f`; R1 remains uncommitted.

## Read these exact checkout-relative paths

1. `AGENTS.md`, `docs/operations/dual-orchestration/README.md`.
2. `docs/operations/dual-orchestration/R1_REVIEW_FINDINGS.md` and
   `docs/operations/dual-orchestration/R1_INGRESS_REPAIR_RESULT.md`.
3. Prior independent verdict:
   `.orchestration/sol-runs/20260905T060349Z-d2435712-4e9a-4af1-a94b-fd183f40c96e/final.txt`.
4. Fresh snapshot: `.orchestration/b0-repair/r1final/review-snapshot.json`.

Verify all fresh hashes before and after review. Only coordination handoff and
active-run state may change separately. Stop on changes in the frozen set.

## Three findings to close independently

| prior finding | required evidence |
|---|---|
| Hostile text `.encode` dispatch before AST parsing | Exact built-in source types only; valid str/bytes pass; subclass sentinel untouched. |
| Proxy/equality/traversal metadata callbacks | Public ingress rejects non-plain JSON before dispatch. Internal immutable views are created from validated copies. No redundant unvalidated metadata argument in source construction. |
| Forged accepted-source AST receipt | Every source-envelope construction derives or revalidates the full receipt from exact bytes, rejecting invalid source with matching fabricated hash/length and valid source with altered AST count. |

- Inspect `src/oracle_composition/rewards/static_validation.py` and the changed
  tests `tests/rewards/test_parent_execution_boundary.py` and
  `tests/rewards/test_scale_calibration.py`; follow their direct consumers only
  as needed. Check valid direct/factory/JSON round trips and deep immutability.
- Public worker start and scale/axiom runtime refusal must remain unchanged.
  The previous broad R1 review covered the child boundary, depth checks, JSON
  numeric checks, deferred tests, and protected constants. Verify those files
  are unchanged against the historical snapshot; do not repeat the full B0/A1
  audit or treat deferred R2/R3 proofs as new R1 failures.
- Report new concrete regressions caused by this repair, if any; distinguish
  ordinary public API paths from deliberate Python object-invariant tampering.

## Safe checks only

- Parent reproduced 143 passes/16 deferred in 2.42s. Verify scope and use
  no-temp tests where helpful; do not claim the parent's checks as yours.
- No candidate compile/exec/invocation, worker subprocess, OS/network canary,
  simulator, training, full suite, package installs, or external research.
- The read-only sandbox may lack temporary storage. Do not retry temp failures
  or weaken isolation. This no-temp selection excludes capture/snapshot tests:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -s -p no:cacheprovider \
  tests/rewards/test_contract.py tests/rewards/test_static_validation.py \
  tests/rewards/test_parent_execution_boundary.py tests/rewards/test_scale_calibration.py \
  -k 'not capture and not snapshot'
```

## Required final

- Start progress/bottleneck/next step; then ACCEPT, ACCEPT-WITH-REPAIRS, or
  REJECT for **R1 static-only scope**, with each prior finding closed or open.
- Exact file/line, mechanism, minimal fix and negative test for remaining defects.
- Checks reproduced versus inspected/deferred; confirm snapshot unchanged.
- No acceptance of B0 dynamic execution, containment, training, or robot
  behavior. If time is short, return partial findings and no acceptance.
