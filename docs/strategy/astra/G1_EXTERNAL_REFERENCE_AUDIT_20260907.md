# G1 external low-gait reference audit

| status | current evidence |
|---|---|
| progress | Three public sources were checked against the current task gate. The first five MotionDecode low-height files all miss `root_z <= 0.50`; one CLAW kinematic reference reaches it; CMU supplies human crouched-walk trials only. |
| bottleneck | No new reference has both resolved rights/schema and evidence of trackability by the frozen 23-DoF GMT controller. MotionDecode has unit/cadence conflicts; CLAW has no visible license; CMU requires a new retargeting path. |
| next step | Keep the eight admitted GMT clips for the current study. Treat external admission as blocked; first request CLAW data terms and its 00454 annotation/cadence reconciliation. Do not screen MotionDecode sample 00006 onward without a new sampling rule. |

## Decision boundary

- **Result:** no external motion is admitted or certified by this audit.
- **Task screen:** a candidate must spend a sustained interval at
  `0.30 <= root_z <= 0.60` and reach `root_z <= 0.50` at least briefly.
- **Method:** public text/source was inspected statically. Numeric CSV bytes were
  streamed through a bounded data-only calculation; no remote file was retained.
  No upstream Python, policy, simulator, retargeter, pickle, or model was run.
- **Speed calculation:** consecutive root `x/y` differences were projected onto
  the normalized root-quaternion local `+x` direction. Reported speeds therefore
  inherit each publisher's position-unit and cadence claims. Summaries use
  steps whose destination frame is inside the task height band.
- **Claim limit:** matching names or deleting wrist columns establishes only a
  proposed field mapping. It does not establish embodiment identity, feasible
  contacts, controller authority, or GMT trackability.

## Fixed GMT target

The external candidates were compared with the admitted GMT source at commit
[`2a590de25a1eb08e47491977a738549c22f16e1f`](https://github.com/zixuan417/humanoid-general-motion-tracking/tree/2a590de25a1eb08e47491977a738549c22f16e1f).

| pinned file | SHA-256 | relevant contract |
|---|---|---|
| [`sim2sim.py`](https://github.com/zixuan417/humanoid-general-motion-tracking/blob/2a590de25a1eb08e47491977a738549c22f16e1f/sim2sim.py) | `10e8cac2e2cd4895ee4b229db4a06aad7f467d1a14399355a7254d789b8e1b5f` | Named 23-joint order; 1 ms simulation step, 20-step decimation, 50 Hz control. |
| [`utils/motion_lib.py`](https://github.com/zixuan417/humanoid-general-motion-tracking/blob/2a590de25a1eb08e47491977a738549c22f16e1f/utils/motion_lib.py) | `c54d8d9f543c6eaf0b9fcd04ad3c8c7447068eece00d83be40741904c1b2f5d1` | Motion payload consumes `fps`, `root_pos`, `root_rot`, and `dof_pos`; source math expects root quaternion `xyzw`. |
| [`assets/robots/g1/g1.xml`](https://github.com/zixuan417/humanoid-general-motion-tracking/blob/2a590de25a1eb08e47491977a738549c22f16e1f/assets/robots/g1/g1.xml) | `7013cd256c89796b2844613d24dda2a13410df7cd32ddd4d59b1741f85094304` | Reviewed GMT G1 model bytes. |
| Local `src/oracle_composition/adapters/gmt/contracts.py` at audit base `455a9d1782656f1af1e529aa9030f79db637f560` | `f574b2a8266d7b4529525f720ce901faa9924538a80288627c558f750ad92f31` | Current 23-DoF and reference-window contract. |

The ordered joint names are left leg
`hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll`; right leg in the
same order; `waist_yaw, waist_roll, waist_pitch`; left then right
`shoulder_pitch, shoulder_roll, shoulder_yaw, elbow`.

## Candidate 1: MotionDecode G1 CSV samples -- structurally close, task-negative

**Source identity.** Hugging Face dataset pin
[`80b489e0378b60bb44d495d5437475ce0e084283`](https://huggingface.co/datasets/CMRobot/MotionDecode/tree/80b489e0378b60bb44d495d5437475ce0e084283),
folder [`1.4.5.Low_Height_Walking`](https://huggingface.co/datasets/CMRobot/MotionDecode/tree/80b489e0378b60bb44d495d5437475ce0e084283/samples/1.4.Constrained_Gait_Category/1.4.5.Low_Height_Walking).
The [README](https://huggingface.co/datasets/CMRobot/MotionDecode/blob/80b489e0378b60bb44d495d5437475ce0e084283/README.md)
(`a637b7cebb2557cc5d55b586a3a93b26bedd81c63ed96521ac867f2643f2ed69`)
describes retargeted Unitree G1 CSV and a dataset-level 120 Hz rate.

**Unresolved units and cadence.** Each CSV header labels root position in
meters, but the README format table says millimeters. The files contain no
timestamp or per-file `fps`. The measurements below assume the header's meters
and the README's dataset-level 120 Hz; durations and speeds are conditional,
not admission facts.

| exact file (direct pinned bytes) | SHA-256 | rows | raw `z_min..z_max` | longest `[.30,.60]` | frames `<=.50` | local forward speed p05 / median / p95, conditional m/s |
|---|---|---:|---:|---:|---:|---:|
| [`CGC_Low_Height_Walking_00001.csv`](https://huggingface.co/datasets/CMRobot/MotionDecode/resolve/80b489e0378b60bb44d495d5437475ce0e084283/samples/1.4.Constrained_Gait_Category/1.4.5.Low_Height_Walking/CGC_Low_Height_Walking_00001.csv) | `0fe1c8cd783aa359cb1f8df6aa4710d8c5b9f14c0ba53a1c5d886a81da3967fe` | 6,412 | `0.505125--0.782777` | 253 frames / 2.108 s | 0 | `-0.039 / 0.301 / 0.552` |
| [`CGC_Low_Height_Walking_00002.csv`](https://huggingface.co/datasets/CMRobot/MotionDecode/resolve/80b489e0378b60bb44d495d5437475ce0e084283/samples/1.4.Constrained_Gait_Category/1.4.5.Low_Height_Walking/CGC_Low_Height_Walking_00002.csv) | `7a3bca1a6179c9cce7dc01c1b4816bc60cb2b4515b89f5d67b7ad1b764fc1f87` | 6,412 | `0.583001--0.772930` | 38 / 0.317 s | 0 | `-0.078 / 0.021 / 0.621` |
| [`CGC_Low_Height_Walking_00003.csv`](https://huggingface.co/datasets/CMRobot/MotionDecode/resolve/80b489e0378b60bb44d495d5437475ce0e084283/samples/1.4.Constrained_Gait_Category/1.4.5.Low_Height_Walking/CGC_Low_Height_Walking_00003.csv) | `2b25ab7c1443777cc9fcaf51ec4d20ab2cb6d6e2029328e837c667c6316d307b` | 6,886 | `0.543221--0.788660` | 101 / 0.842 s | 0 | `-0.004 / 0.448 / 0.584` |
| [`CGC_Low_Height_Walking_00004.csv`](https://huggingface.co/datasets/CMRobot/MotionDecode/resolve/80b489e0378b60bb44d495d5437475ce0e084283/samples/1.4.Constrained_Gait_Category/1.4.5.Low_Height_Walking/CGC_Low_Height_Walking_00004.csv) | `bb65525d90f9373fdeb64913ff509e2f540bd3fe007022c56ec42ca9cf7659c5` | 6,886 | `0.595096--0.774875` | 29 / 0.242 s | 0 | `-0.065 / 0.021 / 0.047` |
| [`CGC_Low_Height_Walking_00005.csv`](https://huggingface.co/datasets/CMRobot/MotionDecode/resolve/80b489e0378b60bb44d495d5437475ce0e084283/samples/1.4.Constrained_Gait_Category/1.4.5.Low_Height_Walking/CGC_Low_Height_Walking_00005.csv) | `eec336177d2f0dc701fd0301ee2bfeefbe3349911ba4f4407d055dd50fbc34dc` | 6,340 | `0.647543--0.781246` | 0 | 0 | n/a |

**Proposed structural mapping.** The CSV header explicitly names 29 G1 joints:
the GMT leg/waist/arm order plus left and right
`wrist_roll, wrist_pitch, wrist_yaw`. Reorder root quaternion `wxyz -> xyzw`
and omit those six wrist fields to obtain the 23 GMT names. This does not prove
that the donor G1 model, root frame, limits, contacts, or generated motion match
GMT.

**Rights/access blocker.** README metadata says `other` / `chingmu-terms` but
links to `LICENSE`, whose exact bytes are empty
(`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`). A separate
[`LICENSE.md`](https://huggingface.co/datasets/CMRobot/MotionDecode/blob/80b489e0378b60bb44d495d5437475ce0e084283/LICENSE.md)
(`ab46dc8e0994ea94fd6f66c0ec0cd33fa3d309702297cc5e220eda58bfc1d60e`)
permits non-commercial academic research, personal study, and prototyping;
requires attribution; and reserves commercial distribution, resale, and
competing hosting. The README also says both fully public and request access for
the complete data. Resolve the linked-license and access contradictions with
the publisher before admission.

**Disposition:** reject these five for the current task gate even under the
favorable meter/120 Hz assumptions. This audit deliberately stops at file 00005.

## Candidate 2: CLAW `00454_crouch` -- best shape, rights-blocked

**Source identity.** Dataset pin
[`cbd516f611e46d52182d261ca1651563d8044850`](https://huggingface.co/datasets/JianuoCao/CLAW/tree/cbd516f611e46d52182d261ca1651563d8044850).
Its [README](https://huggingface.co/datasets/JianuoCao/CLAW/blob/cbd516f611e46d52182d261ca1651563d8044850/README.md)
(`11b5301808f072c97f315109121362c09c473ff2867d7fdee17b81aa1bb69996`)
specifies 50 Hz, root meters, quaternion `xyzw`, and 29 joint radians.

| pinned file | SHA-256 | measured data property |
|---|---|---|
| [`motion/00454_crouch/trajectory_kinematic.csv`](https://huggingface.co/datasets/JianuoCao/CLAW/resolve/cbd516f611e46d52182d261ca1651563d8044850/motion/00454_crouch/trajectory_kinematic.csv) | `c887826193717dd72262311838234f43622917af7340419b3ea39adea94ee6d5` | 296 rows; `z_min=0.476025`, `z_end=0.498190`; 216 consecutive band frames (4.32 s); 88 frames `<=0.50`; local forward full range `0.439--0.720` m/s, p05/median/p95 `0.490/0.564/0.697`. |
| [`motion/00454_crouch/trajectory_dynamic.csv`](https://huggingface.co/datasets/JianuoCao/CLAW/resolve/cbd516f611e46d52182d261ca1651563d8044850/motion/00454_crouch/trajectory_dynamic.csv) | `7880f459daa858c302cb6c90523fad86f4399d25577e67116ad2e96b9921c45d` | 296 rows; `z_min=0.538575`, `z_end=0.576066`; 208 consecutive band frames (4.16 s); zero frames `<=0.50`; local forward full range `0.283--0.878` m/s, p05/median/p95 `0.433/0.643/0.832`. |
| [`motion/00454_crouch/annotation.json`](https://huggingface.co/datasets/JianuoCao/CLAW/resolve/cbd516f611e46d52182d261ca1651563d8044850/motion/00454_crouch/annotation.json) | `f88a1ee138da137ec5fe764072aea032ab8848354ceaf382168477e4fed4b5d0` | Labels one forward crouch, 5.44 s, frames 0--272. This conflicts with 296 rows / 5.90 elapsed seconds at 50 Hz. |

**Named mapping evidence.** The companion
[`g1_xml/g1_29dof_rev_1_0.xml`](https://huggingface.co/datasets/JianuoCao/CLAW/blob/cbd516f611e46d52182d261ca1651563d8044850/g1_xml/g1_29dof_rev_1_0.xml)
has SHA-256 `022cb662e374adb2d2fbf809fa65e92a09b539ed48181b20af12566a0d406429`
and lists the same 29-joint leg/waist/arm/wrist order. The publisher's
[`script/render_video.py`](https://huggingface.co/datasets/JianuoCao/CLAW/blob/cbd516f611e46d52182d261ca1651563d8044850/script/render_video.py)
(`cc8e9fe13db0b83ac11df9d534a956c660e91d7c27e2377e57cc96f4d95b5bf4`)
assigns each 36-value row to that model's `qpos` and reorders only the root
quaternion. Omitting the six named wrist joints therefore gives a deterministic
23-field proposal, not evidence that GMT can track it.

**Dynamics evidence and limit.** The pinned source-repository README at commit
[`92983401d76434870070ff558729e7052db56f63`](https://github.com/JianuoCao/CLAW/tree/92983401d76434870070ff558729e7052db56f63)
(`README.md` SHA-256
`d30dc0e6a4751e63e032e1b18ea97120445105440788eed0e66d1ea54bc91982`)
describes the dynamic export as measured physics-simulated trajectory from its
SONIC-based controller. That is evidence for a different controller/runtime,
not for the frozen GMT tracker; its observed dynamic trajectory also misses
`z <= .50`.

**Rights blocker.** Neither the dataset root nor source-repository root exposes
a license file or dataset license metadata at the pins above. Public readable
bytes are not a reuse grant. Obtain written terms before fetching for admission,
retaining, transforming, or redistributing these files.

**Disposition:** strongest task-shaped kinematic candidate is this file,
but it remains blocked on rights, annotation reconciliation, exact model/root
semantics, and a separately authorized GMT dynamics certificate.

## Candidate 3: CMU Subject 136 -- human-motion retargeting fallback

**Source identity.** The mutable [CMU Subject 136 page](https://mocap.cs.cmu.edu/search.php?subjectnumber=136)
(observed page SHA-256
`d66845af19b829af983ec71cd521481d6fbe97e45ac5b2f5579a26b7e7252870`)
labels trials 09 and 10 `Walk Crouched` at 120 Hz:

| observed source bytes | SHA-256 | source-only duration |
|---|---|---:|
| [`136.asf`](https://mocap.cs.cmu.edu/subjects/136/136.asf) | `48eb3f19477086e28495d841c604157b354fc76be054fa662ff10843ac698427` | skeleton |
| [`136_09.amc`](https://mocap.cs.cmu.edu/subjects/136/136_09.amc) | `2d42440a0cd5abe84a5e4728073425ecfab53787928134536e6e8da84207e79c` | 1,130 frames / 9.408 s |
| [`136_10.amc`](https://mocap.cs.cmu.edu/subjects/136/136_10.amc) | `1d405f6d991acbfef749073ca3caf7e6e347af3d3a60e579509d2424028eee68` | 1,337 / 11.133 s |

These hashes freeze the bytes observed in this audit; CMU supplies no
versioned release pin at those URLs. Human-root height and speed do not answer
the G1 `.30..60` task gate before retargeting.

**Credible but separate conversion project.** GMR commit
[`bb1bbe40774794fceb2a7c579a3464a28e68c844`](https://github.com/YanjieZe/GMR/tree/bb1bbe40774794fceb2a7c579a3464a28e68c844)
documents AMASS SMPL-X to Unitree G1 29-DoF conversion. Its
[`README.md`](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/README.md)
SHA is `06c2752de4f7c0436998e83bb40b54bdc0f6ff3d50c1e73659c65f0278d640a9`;
[`scripts/smplx_to_robot.py`](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/scripts/smplx_to_robot.py)
SHA is `4b4f343dfa040e5e8568042a89e9b8c90eee4a716f391a364904672f435b34a3`.
That script targets 30 Hz, produces `fps/root_pos/root_rot/dof_pos`, converts
root `wxyz -> xyzw`, and writes pickle. The harness must not ingest that pickle;
a reviewed data-only export would be required. Exact survival of trials 136_09
and 136_10 in AMASS SMPL-X was not established.

**Rights.** The [CMU terms](https://mocap.cs.cmu.edu/) (observed page SHA-256
`4ac0024566573a10c87b668c826c2c44014b032ae57cbc5fbef2794b80c21cbc`)
allow research and commercial-product use but prohibit direct resale, including
converted data, and request attribution. The AMASS route is governed by the
[AMASS license](https://amass.is.tue.mpg.de/license.html) (observed SHA-256
`c34c503052154a71e1b196540c5551e92523100ebd9ef172403f61d3a391be26`),
which limits use to non-commercial purposes and prohibits sharing. GMR code is
[MIT-licensed](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/LICENSE),
with license-file SHA-256
`c5a85c0b0012230739a0ab8f30eafba7f8ee7a1b29e025d01e27fa330298e522`,
but that license does not grant rights to AMASS or CMU data.

**Disposition:** semantically relevant source motion, not native G1 data. It is
the highest-cost route and would create a new retargeter/cadence/data family;
consider only after near-native rights/schema routes fail.

## Minimal external-admission gate

1. Human review confirms data rights and retains the exact applicable license.
2. Publisher evidence resolves cadence, root units/frame, G1 model revision,
   column order, and annotation length.
3. A reviewed data-only converter emits a new immutable GMT artifact with
   `fps`, root `xyz`, root quaternion `xyzw`, and the named 23 joints; no pickle
   or executable model crosses admission.
4. Recompute the task screen from the converted bytes. Reject if the strict
   height gate is missed; do not alter the gate to fit a clip.
5. Run a separately authorized frozen-GMT Tier-D probe with survival, all-contact,
   root-height, heading, and joint-tracking metrics. Until then, label it Tier K
   only and make no dynamics or trackability claim.
