# Source-motion registry

| | Status |
|---|---|
| **progress** | Exact DeepMimic humanoid3d source bytes and rights metadata are registered. |
| **bottleneck** | No source motion has a Gymnasium Humanoid mapping or dynamics-feasibility certificate. |
| **next step** | Define and test a controller-native retargeting/admission pipeline before any motion enters training. |

| Source | Current role | Tracker admission |
|---|---|---|
| [`deepmimic_humanoid3d`](deepmimic_humanoid3d/README.md) | Walk, run, face-up get-up, and face-down get-up source candidates | Not admitted |

Source registration preserves bytes and provenance. It does not imply skeleton
compatibility, retargeting correctness, physical feasibility, or useful robot
behavior.
