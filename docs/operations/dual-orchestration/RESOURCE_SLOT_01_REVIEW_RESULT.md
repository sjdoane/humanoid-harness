# Resource slot 01: independent findings and bounded repair

| status | current truth |
|---|---|
| progress | Core token behavior passed static review; both reported error-boundary defects have parent repairs and regressions. |
| bottleneck | Repair closure is not independently accepted. Supervisor integration remains Fable-owned. |
| next step | Review these exact repaired bytes, then hand over the accepted helper; no real compute reservation or training. |

## Independent review

- Run: `.orchestration/sol-runs/20260906T144308Z-40a75826-5de9-4d59-9179-ee8f997f0c37`.
- Requested Sol/max; terminal `SUCCEEDED`, exit 0 at **14:54:15 UTC**.
- Watcher: `terminal_observed`; no deadline signal was needed.
- Verdict: `REPAIRS_RESOURCE_SLOT_01`.
- Final SHA-256: `09c093339a7e20ff1b3020a6295d39e9f6fb54dadfc40747b87ea3b51978ebc9`.
- All original five hashes reverified by reviewer and parent before repair.
  Original identity table: `RESOURCE_SLOT_01_PARENT_RESULT.md`.
- Static only: no reviewer tests or runtime. Core exclusion, authority,
  exact-token release, expiry and full pre-spawn binding passed that review.

| finding | failure | parent repair |
|---|---|---|
| SLOT-01, high | Resolving the supplied root erased its leaf symlink before validation; CLI prepared the target first. | Helper checks lexical leaf with `lstat`; slot CLI commands reject the leaf symlink before resolve/prepare. |
| SLOT-02, medium | NUL authority paths and decoded lone surrogates could emit tracebacks instead of bounded JSON errors. | Reject NUL authority paths; validate UTF-8 strings/keys; translate path/text errors at helper and CLI boundaries. |

## Parent regression evidence

- Parent obtained its own lease after reviewer terminal. Only the already
  accepted helper/CLI/test paths were edited; no supervisor or peer edit.
- New tests on original source: **8 failed, 1 passed, 28 deselected** in 0.40 s.
  The extra surrogate-key case already failed closed; retain it as coverage.
- Final two-file suite: **37 passed in 2.56 s**, with pytest cache and bytecode
  disabled. Ruff, format and diff checks pass.
- Direct helper loading under system Python **3.9.6** resolves this checkout;
  no NumPy, Gymnasium, MuJoCo or Torch imports.
- Root regressions check direct read/reserve/release and all three slot CLI
  commands. Target token bytes, guard bytes/inode and CLI target directory
  contents remain unchanged. Other mailbox command semantics stay unchanged.
- Malformed CLI cases cover NUL authority path and lone surrogates in owner,
  argv, acceptance body and an input key: exit 2, bounded JSON, no traceback,
  token/guard unchanged. Fixtures remain temporary, not the real mailbox slot.

## Repaired bytes for closure

Implementation remains uncommitted until closure. Paths are checkout-relative.

| path | SHA-256 |
|---|---|
| `src/oracle_composition/harness/resource_slot.py` | `a3797f2a15ee7dbca962d6a551382b5e17b796a09b10f82ce73164e693f4a6eb` |
| `tests/harness/test_resource_slot.py` | `0e88d8acd53488c4ee02960916af1025f8872ed4ca36498ad1c046676d8f1514` |
| `scripts/research-mailbox` | `16dbaba3f2f557f6a959337fdafa35366d8d9237bdb27573757f7dbf178456e8` |
| `tests/integration/test_research_mailbox.py` | `11181f94e7bd9cc686e6cc055081683819a94c39c979631e496610d44f70d0c7` |
| `docs/operations/dual-orchestration/HEAVY_JOB_SLOT.md` | `eea27bfc805fb0d7e64aa9522267e1aba0cae16717867d260dd37accd3e7fb63` |

- `uv.lock` unchanged: `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`.
- No simulator, training, full suite, real token, receipt regeneration or push.
- Fable's T2PAIRR2 handoff `20260906T144450.248990Z-ecd1fdd8a16e48d091fd350b2cafec9f`
  was acknowledged as read. Parent confirmed `d97f8da` changes pairing tests
  and records, not runtime; contracts hash still `c25d61de...`. Fable's stated
  combined readback includes this pairing repair and F3 dispatch. Do not
  duplicate that review or infer acceptance from its reported 1,933 passes.
