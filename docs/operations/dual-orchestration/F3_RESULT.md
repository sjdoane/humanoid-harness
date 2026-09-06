# F3: initial-only T2 packet and retained-run ingestion

| status | current truth |
|---|---|
| progress | Implemented the isolated initial-only T2 packet, deterministic prompt, four-file raw retention, envelope validation, and hypothesis-only proposal/recipe receipts. |
| bottleneck | Independent acceptance, registry/admission, the verified stock-COM adapter gate, execution manifest, protected evaluation, candidate measurements, training, and any improvement result remain missing. |
| next step | Run an independent review of this uncommitted four-file slice before any candidate call or integration decision. |

## Builder boundary

| item | observed value |
|---|---|
| checkout | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` |
| branch | `astra/reward-loop` |
| starting HEAD | `f7f3244a910110ddb5a13ffc9ea583892c06d412` |
| starting dirty state | clean (`git status --short` emitted no paths) |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |
| writer lease | owner `astra-f3-builder-20260905`; requested `gpt-5.6-sol`, effort `max`; scope exactly the four F3 files |
| lease acquired | `2026-09-05T23:56:46Z` |
| 20-minute deadline | `2026-09-06T00:16:46Z` |
| implementation verification | `2026-09-06T00:11:53Z`; 15m07s after lease acquisition and before the deadline; no commit or push |

Only these files were added:

- `src/oracle_composition/reward_search/t2_model_contracts.py`
- `src/oracle_composition/reward_search/t2_model_protocol.py`
- `tests/reward_search/test_t2_model_protocol.py`
- `docs/operations/dual-orchestration/F3_RESULT.md`

No existing source, test, configuration, export, CLI, launcher, hook, peer file,
or dependency file was edited.

| new implementation artifact | SHA-256 |
|---|---|
| `t2_model_contracts.py` | `9cd79ea795acce2175e2fc34cfbcb8376f212b2b92a690bed1addbe064918bca` |
| `t2_model_protocol.py` | `0eae5d3b3d91dbb3412393ceb432ca13cbe5fe3a0222bdd82173b4e572f074f8` |
| `test_t2_model_protocol.py` | `66389f92153eb81f31d3fc143e78ded927a8370837945c8341bdb82e793c55e1` |

## Implemented capability

- Accepts only exact builtin baseline bytes matching the 1,144-byte,
  no-newline Git blob at object `a51ea9e68aa9e27a573e372f67e36d6fa4d494cd`.
- Retains and revalidates the baseline plus exact accepted recipe/parser,
  evaluator, T2-input source, canonical input-schema, and 137-byte bounds.
- Builds one strict schema-v3 initial record with no parent recipe, no aggregate
  feedback, target `3.0` m/s, cadence `0.015` s, and only `alpha`/`beta`
  authorable.
- Hashes the canonical semantic request before rendering; exposes the request,
  baseline, T2-contract, and dossier digests in the prompt; records the final
  prompt digest/count afterward without requesting a self-referential echo.
- Reconstructs the complete record and all immutable local identities at every
  render, publish, and ingestion boundary. Exact byte and native-Path types are
  required before caller-controlled callbacks or publication.
- Bounded-reads and no-overwrite-retains each available `request.json`,
  `task-packet.md`, `result.json`, and `final.txt` before parsing. Missing,
  empty, oversized, nonregular, and unreadable inputs have explicit unretained
  states without fabricated hashes or counts.
- Accepts only a locally consistent read-only Sol/max screen envelope owned by
  `astra-f3-initial`, with candidate role, fixed scope, matching request/result
  metadata, exact prompt identity, un-escalated exit 0, a thread ID, and a
  terminal time within 1,200 seconds of immutable request creation.
- Uses the existing trusted parser to turn accepted `alpha`/`beta` into the
  fixed formula recipe. The response cannot choose formula/parser/runtime IDs,
  code, paths, execution, retries, or evidence.
- Emits schema-isolated receipts distinguishing local envelope consistency from
  authenticated model origin. Acceptance is format-only and hypothesis-only.

## Frozen input identities

| artifact | bytes / SHA-256 |
|---|---|
| exact tracking-only baseline | 1,144 / `0986d4fc907e94185e14f58c9ef6eabf8ec26a8f336c68ee450bbacd5d8224d4` |
| recipe/parser source | `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb` |
| F2 evaluator source | `064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301` |
| T2 input source | `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2` |
| canonical T2 input schema | 396 / `8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9` |
| canonical F2 bounds | 137 / `2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933` |

The baseline is preserved as a declared tracking-only configuration, not an
observed successful load. `alpha=1,beta=0` is not treated as its parent, and
`alpha=0` remains outside the accepted formula family.

## Verification

| check | result |
|---|---|
| new F3 tests | `35 passed in 0.26s` |
| new F3 plus permitted F1 regression | `79 passed in 0.47s` in the final focused run |
| Ruff lint, three Python files | passed |
| Ruff format check, three Python files | passed |
| `py_compile`, three Python files | passed |
| no-index `git diff --check`, all four new files | passed |

The tests use real temporary mode-0600 fixture files and the production bounded
reader, parser, and no-overwrite publisher. They cover the six requested groups:
valid raw retention, altered envelopes and file states, immutable-context
refusals, malformed/typed/nonfinite response rejection, exact boundary and
callback ordering, and A1/F1/T2 schema isolation.

No model candidate call, subprocess launcher, simulator, training, network,
dependency install, full suite, or fixture publication outside pytest temporary
directories was run.

## Unresolved issues and claim ceiling

- Independent review and acceptance have not occurred. A successful builder and
  green focused checks do not authorize a candidate call or integration.
- Candidate reward admission and the verified inclusive `[-25,25]` m/s stock
  COM origin/cadence adapter gate remain missing.
- No execution manifest, protected evaluator result, candidate measurement,
  registry entry, runtime authorization, training authorization, or measured
  improvement exists.
- The exact canonical bounds are locally bound, but shared promotion remains a
  separate peer/owner decision.

**Claim ceiling:** initial-only packet construction, bounded local run-artifact
retention, deterministic envelope-consistency validation, and parameter-recipe
format acceptance. Nothing here establishes authenticated model service,
physical admission, simulator behavior, training, reward improvement, or
humanoid competence.
