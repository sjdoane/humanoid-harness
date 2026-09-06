# GMT source attribution

- Upstream: [Humanoid General Motion Tracking](https://github.com/zixuan417/humanoid-general-motion-tracking/tree/2a590de25a1eb08e47491977a738549c22f16e1f).
- Pin: `2a590de25a1eb08e47491977a738549c22f16e1f`.
- `reference_math.py`, `reference_runtime.py`, and the deployment ABI adapt
  `utils/torch_utils.py`, `utils/motion_lib.py`, and `sim2sim.py` at that pin.
- Changes: no downloaded Python execution or pickle/JIT loading; explicit
  numeric artifacts, validation, bounded headless execution, and retained traces.
- The upstream Apache-2.0 text is retained in `LICENSE-GMT`.
- This code attribution does not grant redistribution rights for weights,
  meshes, or motion data. Those assets remain ignored, local research inputs.
