# F2: accepted pure T2 evaluator

| status | current truth |
|---|---|
| progress | Independent static review accepts F2; parent reverified all 13 snapshot hashes and reproduced 73 focused passes. |
| bottleneck | Registry promotion and adapter admission are not implemented by this slice; T2-aware model generation and training remain gated. |
| next step | Hand Fable the exact reviewed artifact identities and resolve the canonical bounds/admission details before promotion. |

## Acceptance evidence

- Verdict: `ACCEPT_T2_EVALUATOR_ONLY`; no blocking findings.
- Review run: `20260905T224027Z-d05d46fb-4723-4448-b1dd-39a3e12d8934`.
- Terminal: `SUCCEEDED` at 22:45:28 UTC; requested read-only Sol/max,
  no escalation; deadline watcher observed completion.
- Reviewed base: `a0d976bdbe0bb7fc6464611d9ae14c64681d4481`.
- Snapshot: `.orchestration/f2-review-20260905T2237.json`; root, HEAD, lock
  and 13/13 file identities matched before/after review and parent collection.
- Reviewer regenerated bounds in memory and inspected the real validation,
  arithmetic, callback tests and exact dependency bindings.
- **Reviewer pytest: 0 tests.** Read-only pytest failed before collection
  because it could not create a temporary directory. This is not an
  independently reproduced test pass; no permissions were bypassed.
- Parent previously ran 73 passes in 0.04 s and repeated **73 passes in
  0.07 s** at this acceptance checkpoint. Ruff/format/compile checks were
  completed before review; no source changed afterward.

## Exact handoff

| field | value |
|---|---|
| runtime ID | `target_speed_triangular_affine_t2_adapter/v1` |
| evaluator file | `src/oracle_composition/rewards/target_speed_formula_t2.py` |
| evaluator SHA-256 | `064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301` |
| parser ID | `target_speed_triangular_affine_recipe/v1` |
| parser callable/file | `parse_target_speed_formula_recipe`; `src/oracle_composition/rewards/target_speed_formula.py` |
| parser/source SHA-256 | `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb` |
| internal recipe ID | `target_speed_triangular_affine/v1` |
| T2 input source SHA-256 | `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2` |
| bounds SHA-256 | `2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933` |
| output | Exact finite builtin float in `[-10.0,15.0]` |

Canonical bounds are these **137 bytes**, with no trailing newline:

```json
{"parameters":{"alpha":{"maximum":4.0,"minimum":0.25},"beta":{"maximum":10.0,"minimum":-10.0}},"r_task":{"maximum":15.0,"minimum":-10.0}}
```

## Peer conditions and claim boundary

- Fable accepted the ID/parser interface in
  `20260905T230846.027071Z-a969235f97034a998d56358ecec842a6`, replying to
  proposal `20260905T220539.436910Z-95e3e2a4b5954edc81e2bfe32f171cd7`.
- Its reply described bounds with interval arrays. Promotion must confirm the
  exact canonical object/hash above; equivalent intervals are not identical bytes.
- Keep the accepted T2 class unchanged. Fable's proposed adapter certificate
  will require COM speed in inclusive `[-25,25]` m/s, verified stock COM
  measurement origin and 0.015 s cadence. Out-of-range input is rejected, not
  clipped. This is a runtime admission obligation, not part of the pure API.
- This slice does not implement or certify that certificate. General finite
  inputs remain intentional for pure arithmetic and refusal testing.
- No shared registry entry, compositor change, live-model origin, simulator,
  trained cycle or behavioral improvement is established here.
