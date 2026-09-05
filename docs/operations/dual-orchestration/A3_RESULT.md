# B0 adoption: retained leaf stop and parent verification

| status | result |
|---|---|
| progress | Parent verified and adopted 25 exact new B0 files for repair. No missing predecessor source dependency was found; A1 and all existing files were preserved. |
| bottleneck | The leaf audit stopped on an ambiguous handoff path. B0 remains execution-blocked by its existing P1 findings. |
| next step | Dispatch R1: remove parent-side candidate execution, enforce parse-only validation, and repair the two pure input-validation defects. |

## Retained leaf outcome

- Run: `20260905T022721Z-6b3b7fa8-1a3a-456a-b63d-ff67ab4614d5`.
- Terminal: `SUCCEEDED` at 02:28:15 UTC; audit **not completed**.
- The packet said `ASTRA_HANDOFF.md` without an unambiguous full path. The
  worker looked in the checkout root, found no file, and stopped as instructed.
- Correct location: `docs/operations/dual-orchestration/ASTRA_HANDOFF.md`.
- The parent completed this narrow check locally; no duplicate audit call.
  The original packet/output stay intact. New packets use explicit paths.

## Parent adoption checks

- Donor: `89d4c34d1b04eb9efca2b75d1ffc828a13de55d7`.
- Astra base: `96720e577727387d8f32b117f46786872a3d56c5`; clean before adoption.
- Read both complete B0 reviews and the transfer message. Inspection confirms
  parent execution in static validation, source capture, and scale probes.
- Between Astra's `05c0469` fork and B0, all existing source, dependency-lock,
  project metadata, and test-conftest bytes are unchanged. Only the nine B0
  modules are added in those paths; intervening commits record operations.
- Non-B0 imports required by the manifest already exist unchanged:
  `envs/humanoid.py` (`CONTACT_CAPTURE_ID`, `HumanoidExperimentConfig`,
  `make_humanoid_env`) and `experiments/runtime_identity.py`
  (`dependency_lock_path`, `module_sha256`, `source_tree_sha256`, `space_sha256`).
- Adopted the 25 transferred new source/test/experiment files with
  `apply_patch`. Every target was absent. Every resulting Git blob matches
  the donor exactly; per-file SHA-256 values are in `B0_DONOR_SNAPSHOT.json`.
- Excluded ADR 0007, decisions index, and Fable's main handoff. A whole-commit
  cherry-pick would mix those shared operational changes into this repair.
- No dirty main files, environments, hooks, or leases were copied or changed.
- Post-adoption A1/mailbox regression: `59 passed in 2.23s`.
- Diff check reports six inherited Markdown hard-break spaces in the donor
  protocol/decision rule. Preserved exact donor bytes; new coordination docs
  pass the scoped whitespace check. No donor reward tests were executed.
- Donor receipts remain historical evidence, not fresh Astra certification.
  The whole-package fingerprint cannot replay unchanged after A1 adds files.
  Do not regenerate it merely to conceal that coupling.

## Repair order

| slice | scope | existing findings |
|---|---|---|
| R1 | Parent parse-only boundary; child-only installation; reject arbitrary calibration callables; conditional depth; strict JSON types | SCI-02/06, ROB-01/10; remove unmeasured compile work from source checks |
| R2 | Production worker containment, categorical failures/deadlines, real denial controls, load-time identity and capability evidence | SCI-03, ROB-03/04/05/06/07/09 |
| R3 | Admission/affine and episode lineage, scoped runtime fingerprints, receipt ledger, remaining negatives and rights ingress | SCI-01/04/05/07/08, ROB-02/08/11/12/13, Fable fingerprint finding |

- Repair source locally first. All P1s and their independent re-review remain
  prerequisites for real generated-candidate validation or training.
- R1 cannot claim a working sandbox or complete dynamic validation. Its
  verification uses data/static tests only; no candidate source execution.
- R2 requires a separate bounded host-fixture protocol before live containment
  tests. R3 shared protocol/ADR/config and external-rights edits need explicit
  peer scope agreement. No heavy/full-suite job without resource agreement.
