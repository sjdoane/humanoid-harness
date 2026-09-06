# F3: initial-only T2 packet and retained-run ingestion

- One Sol/max builder; no delegation. Deadline: 20 minutes from launch.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`,
  branch `astra/reward-loop`. Record starting HEAD, dirty state and lock hash.
- Use its writer lease and `apply_patch`; no commit/push. Only four new files:
  1. `src/oracle_composition/reward_search/t2_model_contracts.py`
  2. `src/oracle_composition/reward_search/t2_model_protocol.py`
  3. `tests/reward_search/test_t2_model_protocol.py`
  4. `docs/operations/dual-orchestration/F3_RESULT.md`
- Do not edit existing source/tests/config/exports/CLI/launcher/hooks or peer
  files. No model candidate call, subprocess launcher, simulator, training,
  network, dependency install, full suite or fixture publication outside tests.

## Read

- Root `AGENTS.md`, dual-orchestration `README.md`, `F3_PLAN_RESULT.md`,
  `F2_ACCEPTANCE.md`, `F1_REVIEW_FINDINGS.md` (last three in that docs directory).
- Relevant packet/envelope/retention sections of existing reward-search
  `contracts.py`, `loop.py`, `formula_contracts.py`, `formula_loop.py`,
  `publication.py`, and accepted reward formula/T2 modules.
- The exact baseline Git blob identified below. Use `git show` in Astra;
  never read a peer dirty file or import its simulator-facing package.

## Minimal public API

```python
def prepare_initial_t2_packet(*, baseline_reward_bytes: bytes) -> bytes: ...
def render_initial_t2_prompt(record_bytes: bytes) -> bytes: ...
def publish_initial_t2_packet(record_bytes: bytes, packet_path: Path, record_path: Path): ...
def ingest_initial_t2_sol_run(
    record_bytes: bytes, run_directory: Path, records_directory: Path
): ...
```

- Exact builtin bytes only at record boundaries; exact native Path objects at
  path boundaries. Reject subclasses/typed Pydantic objects before callbacks.
- Use strict frozen Pydantic models internally: distinct T2 initial record,
  request, parameter proposal, run-artifact binding and model-call receipt.
  Required literal kind/version fields; forbid extras and nonfinite numbers.
- Bound packet/record bytes at 131,072, final response at 65,536 and each
  request/result JSON at 65,536. Normalize duplicate-key, encoding, malformed
  JSON, nesting/recursion and validation failures into a small protocol error.
- Reuse A1 `_read_bounded_regular_file`, `parse_model_bytes`, detached
  request/result models and no-overwrite publication helpers unchanged.
  Add the exact-type wrapper before using A1 parsing. Reuse generic retained
  artifact support where it helps; do not copy F1's loop or its schema family.

## Exact initial context

- Baseline must match **exactly**, including its no-newline bytes:
  `a51ea9e68aa9e27a573e372f67e36d6fa4d494cd:experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json`;
  1,144 bytes, SHA-256
  `0986d4fc907e94185e14f58c9ef6eabf8ec26a8f336c68ee450bbacd5d8224d4`.
- It is declared tracking-only configuration, not an observed successful load.
  Preserve its exact contents and internal identities. No parent recipe:
  alpha=1,beta=0 is not the baseline; alpha=0 is outside the formula family.
- Retain/bind exact local accepted source bytes for the recipe/parser
  (`c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb`),
  F2 evaluator (`064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301`),
  T2 inputs (`9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2`),
  canonical T2 schema (`8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9`),
  and 137-byte F2 bounds (`2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933`).
- Target exactly 3.0 m/s, cadence metadata 0.015 s, only alpha/beta authorable.
  Recipe/parser/runtime IDs are those in `F2_ACCEPTANCE.md`.
- Fixed initial dossier: no aggregate feedback; explicitly missing candidate
  reward admission, verified [-25,25] COM adapter/origin/cadence gate, execution
  manifest, protected evaluator results and candidate measurements. Never use
  dummy hashes for missing evidence or claim current physical admission.
- A stale prose baseline hash exists; the exact blob above wins. Initial
  protocol supports this pinned baseline only, not arbitrary new runtime claims.

## Packet, response and ingestion

- Hash canonical request payload before rendering. Show its digest, baseline
  digest, T2-contract digest and evidence-dossier digest explicitly in prompt.
  Compute final prompt digest/count afterward for the record/launcher; never
  ask the model to echo a self-referential final-prompt hash.
- Reconstruct/revalidate all immutable semantics and digests when rendering or
  ingesting record bytes. Mutated nested records/hashes must not become trusted
  just because a caller reserializes them. Exact initial evidence stays missing.
- Response: unique kind/version; echo those four visible digests; exactly
  `parameters:{alpha,beta}`, bounded rationale/predicted effect/falsifier and
  literal `hypothesis_only_not_admitted`. No Python source or candidate-selected
  formula/parser/runtime fields. Trusted code supplies the fixed recipe ID
  and invokes existing `parse_target_speed_formula_recipe` as sole parser.
- Expected local launcher configuration: read-only `gpt-5.6-sol`, effort `max`,
  screen runner, owner `astra-f3-initial`, role `candidate`, scope
  `t2-initial-parameter-hypothesis-only`. Retained task-packet bytes must match
  the rendered prompt; request digest/count and shared request/result metadata
  must agree. Terminal must be un-escalated success with exit 0 and thread ID.
- For each of the four known run files (request.json, task-packet.md,
  result.json, final.txt), retain each successfully bounded-read exact byte
  sequence with SHA-256/count using no-overwrite publication **before parsing**.
  Fixed safe names/run ID; no candidate-controlled output paths. Missing,
  empty, oversized, nonregular or unreadable files yield an explicit rejection
  state. Never claim bytes rejected by the reader were retained/hashed.
- Valid response yields canonical parameter proposal and recipe artifacts;
  invalid responses keep retained raw bytes and produce a rejection receipt.
  Malformed input record/path fails before publication instead.
- New receipt kind/source class must not cross-parse as A1 or F1. Bind all
  present envelope artifacts, original packet, semantic request and baseline;
  proposal/recipe hashes only for format acceptance. Emit literal metadata
  semantics `requested_configuration_not_served_model_attestation`, with local
  envelope consistency distinguished from authenticated model origin.
- Keep registry/admission missing, runtime/training not authorized and
  improvement not measured in every outcome. No retry, model launch, revision,
  runtime call, optional execution switch or fake feedback in this slice.

## Six focused acceptance groups

1. True baseline packet and valid local fixture envelope; exact raw retention,
   canonical recipe, visible echoed identities and honest hypothesis receipt.
2. Altered packet/envelope identities, failed/interrupted terminal, missing or
   invalid final file: rejection with accurate retained/missing states.
3. Wrong baseline/target/cadence/dependency hash, fabricated parent or fabricated
   evaluator evidence: refusal before admission.
4. Malformed/extra/duplicate keys, bool/string/nonfinite/out-of-bounds params,
   unknown runtime/source-class fields: raw response retained and rejected.
5. Byte/Path subclasses, forged/partial typed records, mutated nested data and
   callback traps: no callbacks or publication before record validation.
6. A1/F1/T2 receipt isolation and unchanged accepted sources. Use only the new
   test file plus existing `tests/reward_search/test_formula_proposal_loop.py`
   for regression if within the deadline. No simulator tests or broad suites.

Test with Astra's own `.venv`, real temporary fixture files and actual helpers,
not mocks that bypass validation. Run new-file Ruff/format/compile and diff
checks. Record exact checks, source hashes, unresolved issues and claim ceiling
in F3_RESULT. A successful builder is not acceptance or candidate-call authority.
