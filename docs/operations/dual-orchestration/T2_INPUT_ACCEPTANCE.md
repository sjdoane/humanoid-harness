# T2 input-only acceptance

| status | current truth |
|---|---|
| progress | Independent review returned ACCEPT_T2_INPUT_ONLY; parent and reviewer each passed all 17 pure tests. |
| bottleneck | Fable has not yet acknowledged the exact interface proposal. Formula adaptation, compositor and runtime measurement remain separate work. |
| next step | Fable confirms the input interface, then pins these exact bytes in its runtime packet. |

- Review: `.orchestration/sol-runs/20260905T193917Z-382d4a3e-5e85-4229-9d82-a7958035231b`.
- Finished: 2026-09-05 19:48:21 UTC; requested Sol/max, read-only.
- Source: `src/oracle_composition/rewards/task_inputs_v2.py`.
  SHA-256 `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2`.
- Test: `tests/rewards/test_task_inputs_v2.py`.
  SHA-256 `1dbc1fa194effd2c84d470dde8a590828911c8c79ae6cbc245f0384c17add63b`.
- Parent reverified the full nine-file review snapshot after terminal status;
  both accepted files are unchanged. Ruff and formatting passed.
- API: `CandidateTaskInputsV2(com_x_velocity_m_s, target_speed_m_s)`;
  `validate_task_inputs_v2` revalidates at consumption.
- Exact builtin finite numbers; fixed target 3.0 m/s; stock COM speed semantics;
  cadence metadata 0.015 seconds. Old B0 inputs and targets remain unchanged.
- Exact-path peer proposal:
  `20260905T193614.811073Z-cc47aa7840274926956386766cdb5c05`.

No formula binding, total reward, model-origin evidence, simulator observation,
training admission or robot improvement is established by this acceptance.
