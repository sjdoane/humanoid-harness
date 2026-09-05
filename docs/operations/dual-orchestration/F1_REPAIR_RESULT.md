# F1 lineage repair result

| status | current truth |
|---|---|
| progress | Repaired only F1-01, F1-02, and F1-03 in the two formula-lineage modules and their focused test file. Final bounded checks pass. |
| bottleneck | This uncommitted slice is not independently closed or accepted, and it remains static/synthetic proposal plumbing rather than reward-runtime or humanoid evidence. |
| next step | Parent performs independent closure against the retained pre-repair snapshots and this receipt; no integration is authorized by this result. |

## Checkout and pre-repair gate

| item | observed value |
|---|---|
| checkout realpath | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` |
| branch | `astra/reward-loop` |
| repair HEAD | `ffb89dd386d0d96619b92087c13efa4a74b7638a` |
| writer setting | One user-requested Sol/max writer; this session exposed no independent served-model attestation |
| delegation/model calls | No delegation, nested worker, paid API call, or additional model call |
| starting dirty state | The seven uncommitted F1 builder paths listed in `F1_RESULT.md`; no tracked modification |
| review receipt | `.orchestration/f1-review-20260905T1937.json` |
| review final | `.orchestration/sol-runs/20260905T193917Z-382d4a3e-5e85-4229-9d82-a7958035231b/final.txt` |
| pre-repair snapshots | `.orchestration/f1-pre-repair-20260905/`; unchanged |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |

Before repair, all seven F1 file hashes and `uv.lock` matched the review receipt.
After repair, the four F1 files outside the authorized repair set still match:

| unchanged path | SHA-256 |
|---|---|
| `experiments/family_b_target_speed_formula_v1/config.json` | `81a08a4e02dbc05da640c4bf0a1a6c1f75a7c4c6b2cbc27282f9feb6bec1e050` |
| `src/oracle_composition/rewards/target_speed_formula.py` | `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb` |
| `tests/rewards/test_target_speed_formula.py` | `a9d124c9616b66cab75175c768028a8a1e1b7c25462b90c9767e29f2d110fe1d` |
| `docs/operations/dual-orchestration/F1_RESULT.md` | `a884802c669d2c7a719d0521002d3a7193b82e18d43f505351a2f7595017b69b` |

## Repairs

| finding | repaired capability | refusal/replay boundary |
|---|---|---|
| F1-01 | Every exact nonempty supplied response of at most 65,536 bytes is locally published once as raw bytes before parsing, including malformed and rejected responses. The receipt binds the artifact filename, recomputed SHA-256, and byte count. | Replay opens the retained single-link `0600` regular file without following symlinks, reads within the declared bound, recomputes byte count and SHA-256, and refuses metadata, content, or receipt mismatch. Revision reparses those retained bytes and requires the resulting canonical proposal to equal the retained proposal. |
| F1-02 | The canonical semantic request payload is formed and hashed before prompt rendering. Its `request_payload_sha256` is explicitly present in the final prompt and required response schema. The separately named `rendered_prompt_sha256` and `rendered_prompt_byte_count` bind exact final prompt bytes. | Packet rendering and ingestion recompute both identities. Revision revalidates the packet and requires the receipt and iteration to retain both semantic-request and exact-prompt identities. The prompt-only responder regression reads every required identity from prompt bytes. |
| F1-03 | Public model-consuming boundaries first require exact trusted classes, preflight nested values without invoking user objects, round-trip the full payload through strict Pydantic validation, and then operate on the returned canonical copy. | Wrong kind/schema, partial model storage, forged nested records, shallow list mutation, and callback-bearing nested values fail closed. Existing per-binding, per-manifest, response-lineage, revision-lineage, schema-separation, and overwrite negatives remain. |

Raw response bytes are retained as data only. They are never executed, imported,
compiled, evaluated, sent externally, or treated as model-call evidence.

## Regression coverage

The formula-loop file now collects 43 cases, up from the builder result's 32.
The 11 new cases cover:

1. a responder that constructs its accepted response from prompt-visible data only;
2. semantic request identity mutation;
3. rendered prompt SHA-256 mutation;
4. rendered prompt byte-count mutation;
5. malformed rejected-response byte retention and replay;
6. retained-byte mutation refusal in replay and revision;
7. raw-response publication collision without overwrite;
8. wrong top-level and nested kind/schema labels;
9. partial receipt lineage;
10. mutated lists and forged nested entries; and
11. callback-bearing nested data refused before callback access.

Existing accepted-ingestion coverage now checks exact response bytes and receipt
bindings. Existing revision coverage now also rejects a forged prompt identity.

## Bounded verification

Every test, lint, format, and compile command ran through a 60-second
`subprocess.run` timeout and used the Astra `.venv`. Pytest cache creation was
disabled, and pytest temporary data stayed under
`/private/tmp/humanoid-f1-repair-20260905/`.

| scope | final result | skips |
|---|---:|---:|
| `tests/reward_search/test_formula_proposal_loop.py` | 43 passed in 0.32 s | 0 |
| formula core plus accepted T2: `tests/rewards/test_target_speed_formula.py tests/rewards/test_task_inputs_v2.py` | 44 passed in 0.03 s | 0 |
| all existing `tests/reward_search` | 93 passed in 0.65 s | 0 |
| Ruff check on the three changed Python files | passed | 0 |
| Ruff format check on the three changed Python files | 3 files already formatted | 0 |
| `py_compile` on the three changed Python files | passed | 0 |

The 93-test reward-search command includes the 43 formula-loop cases. Across the
two non-overlapping final suite commands, 137 tests passed and none skipped. No
full suite was run.

## Import origin

| import | resolved path |
|---|---|
| `oracle_composition` | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/__init__.py` |
| `oracle_composition.reward_search.formula_contracts` | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/reward_search/formula_contracts.py` |
| `oracle_composition.reward_search.formula_loop` | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/reward_search/formula_loop.py` |

## Exact repair diff

Only these paths differ from the retained pre-repair state or are newly added:

1. `src/oracle_composition/reward_search/formula_contracts.py`
2. `src/oracle_composition/reward_search/formula_loop.py`
3. `tests/reward_search/test_formula_proposal_loop.py`
4. `docs/operations/dual-orchestration/F1_REPAIR_RESULT.md`

No snapshot, old receipt, historical builder result, numerical formula/config,
A1/R1 source, accepted T2 source/test, shared runner, compositor, CLI,
dependency, frozen experiment, other documentation, or Git state was changed.
No host probe, reward subprocess, simulator, training, network, install, paid
API call, additional model call, external write, push, commit, or nested worker
ran.

## Remaining limitations

- The response origin remains `unverified_synthetic_supplied_bytes`; no model
  identity, provider attestation, or model-call receipt exists.
- The repair establishes local immutability, lineage, replay, and refusal
  behavior only. It does not establish formula quality or humanoid competence.
- The total-reward compositor remains explicitly absent. No generated Python,
  T2 adaptation, total-reward runtime, calibration, simulator evaluation,
  training, or behavioral comparison is admitted.
- The COM-x versus root-x speed/target mismatch and the earlier R2/R3 work stay
  unresolved and outside this repair.
- F1 remains uncommitted and requires the parent's independent closure before
  any integration decision.
