# Test matrix — 2026-09-06

| status | evidence |
|---|---|
| progress | G1-focused checks pass; whole-repository discovery now collects both reference-runtime test modules. |
| bottleneck | Older native-study tests depend on host-local assets, checkout identity, and historical receipts. The whole suite is not green. |
| next step | Preserve runtime admission gates; separate portable fixtures from explicit host-reproduction checks. |

## Observed runs

| Command / source | Result | Interpretation |
|---|---|---|
| `pytest -q`, clean `1b501d6` | Collection stopped: duplicate module basename and missing `h5py` | No complete test run |
| Full suite with importlib discovery and locked sources extra | 2,111 passed; 37 skipped; 90 failed; 13 setup errors; 277.29 s | Failing integration snapshot, not a passing release |
| G1 + feedback + project CLI after test repairs | 209 passed; 1.67 s | Active adapter checks only |

- Installed `h5py==3.16.0` from the existing lock/cache; no lock change.
- Discovery uses `--import-mode=importlib`; qualified the shared G1 fixture import.
- A data-only test now rejects attempted MuJoCo imports without assuming that
  unrelated earlier tests have never loaded the module.
- Test repairs do not change simulation, rewards, training, or evaluation.

## Remaining failure groups

- Three native corpus tests: ignored Farama actor payload absent in Astra.
- Native no-learning receipt: recorded source-tree fingerprint differs.
- F3 model-protocol tests: live import-origin guard expects the Fable checkout.
- T2 artifact/report tests: old injected admission HEAD conflicts with newer
  descendant-aware experiment state.
- These groups require fixture/host-reproduction review. Do not bypass
  production origin, artifact identity, or execution-seal checks to make tests pass.
- Missing optional source payloads are not evidence that G1 assets are absent:
  G1 uses a separate pinned numeric motion/controller library.
