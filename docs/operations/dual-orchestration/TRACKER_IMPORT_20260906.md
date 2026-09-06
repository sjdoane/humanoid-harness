# Tracker import: exact handoff for independent review

| status | current truth |
|---|---|
| progress | Formal swap accepted; complete FT2R1–FT2R3 chain imported into Astra's checkout. Parent focused checks pass. |
| bottleneck | Combined reviews pending; one unexplained full-suite CLI failure; no trained tracker or calibration result. |
| next step | Run OT1 scientific and robustness source reviews, then resolve findings before requesting the bounded smoke. |

## Transfer receipt

- Fable acceptance: `20260906T040734.663495Z-406cc48ada0f4dfca0a07c315b285b61`.
- Source handoff: `20260906T041335.021200Z-21a1019505f143fcae092881192e36c1`.
- Peer exact commit: `8d617e30dd239529a42e3a0211314d6173ffeafd`.
- Astra import: `851e16d3b274e21a3acb8b5446519bca4d0fd811`.
- No merge conflicts. Source, tests and experiment artifacts match the peer
  exactly; only three Astra coordination documents differed at import.
- Import is for review, not repair acceptance or runtime authorization.
- Four original reports retained byte-for-byte in ignored
  `.orchestration/ot1-review-20260906/`; snapshot records their source paths
  and SHA-256 plus the frozen review input bindings.
- Dispatch packets: `TASK-OT1-scientific-review.md` and
  `TASK-OT1-robustness-review.md`. Durable run pointers:
  `.orchestration/astra-active-run.json`.

## Parent checks

| check | observed result | ceiling |
|---|---|---|
| F3 proposal and formula tests | 112 passed, 0.57 s | Static reward contracts |
| Reference-runtime tests, first attempt | Collection blocked by missing Gymnasium | Environment dependency, no tests ran |
| Environment repair | Offline locked sync with dev/gym/train extras; 22 cached packages installed; lock unchanged | Separate Astra venv only |
| Import origin after sync | `humanoid-harness-astra/src/oracle_composition/__init__.py` | No peer-checkout import |
| Reference-runtime rerun | 6 passed, 6.50 s | Window/phase mechanism with synthetic state; no simulator |

## Unresolved evidence

- Fable full host suite: CLI fake-runtime report test failed, then passed alone.
  Exact test and peer record are in the robustness packet. Cause unknown;
  neither test-order nor host load has been established as the cause.
- No simulator, training, full suite, reward candidate call or receipt
  regeneration launched by Astra during import.
- Smoke still needs the exact peer resource reservation; cohort still needs
  Samuel's explicit authorization. Calibration remains non-scoring.
- Phase A controller-switching results do not show successful numeric-reference
  composition. Three failed handover designs are not universal infeasibility.
- Fable owns reward/feedback and main promotion. Astra owns oracle/tracker
  and both combined reviews; checkouts and writer leases stay separate.
