#!/usr/bin/env python3
"""Render a validated GMT replay trace without rerunning policy or dynamics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.adapters.gmt.contracts import (
    GMT_UPSTREAM_COMMIT,
    SIMULATION_DT_SECONDS,
)
from oracle_composition.adapters.gmt.io import (
    GMTAdmissionError,
    sha256_file,
    write_json_receipt,
)
from oracle_composition.adapters.gmt.replay import (
    ModelABI,
    _extract_model_abi,
    validate_model_abi,
)
from oracle_composition.adapters.gmt.trace_admission import load_validated_replay

FRAME_WIDTH = 480
FRAME_HEIGHT = 360
OUTPUT_FPS = 20
LABEL = "Reconstructed GMT baseline / no learning"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _publish_gif(images: list[Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".gif", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        images[0].save(
            temporary,
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / OUTPUT_FPS),
            loop=0,
            optimize=False,
            disposal=2,
        )
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise GMTAdmissionError(f"refusing to overwrite GIF: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def render_recorded_trace(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    abi: ModelABI,
    arrays: dict[str, np.ndarray],
    support_files: dict[str, str],
    upstream_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.suffix.lower() != ".gif":
        raise ValueError("GMT visualization output must use the .gif suffix")
    receipt_path = output.with_suffix(".gif.manifest.json")
    if output.exists() or receipt_path.exists():
        raise GMTAdmissionError("refusing to overwrite existing GMT visualization")
    try:
        import mujoco
        import PIL
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("MuJoCo and Pillow are required only for approved rendering") from exc

    model = mujoco.MjModel.from_xml_path(str(upstream_root / "assets/robots/g1/g1.xml"))
    actual_abi = _extract_model_abi(mujoco, model)
    validate_model_abi(actual_abi)
    if actual_abi != abi:
        raise GMTAdmissionError("render-time model ABI differs from the replay manifest")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -15.0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    frame_stride = round(1.0 / (OUTPUT_FPS * SIMULATION_DT_SECONDS))
    frame_indices = np.arange(0, arrays["sim_time"].shape[0], frame_stride, dtype=np.int64)
    images: list[Any] = []
    try:
        for index in frame_indices:
            data.qpos[:] = arrays["sim_qpos"][index]
            data.qvel[:] = arrays["sim_qvel"][index]
            data.time = arrays["sim_time"][index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render()).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, FRAME_WIDTH, 66), fill=(0, 0, 0, 178))
            time_value = float(arrays["sim_time"][index])
            control_index = int(
                np.clip(
                    np.searchsorted(arrays["control_time"], time_value, side="right") - 1,
                    0,
                    arrays["control_time"].shape[0] - 1,
                )
            )
            lines = (
                LABEL,
                f"motion: {manifest['inputs']['motion_name']}   t={time_value:0.2f}s",
                "root-height proxy: "
                f"{arrays['sim_qpos'][index, 2]:0.3f} m   "
                f"joint RMSE: {arrays['control_joint_rmse'][control_index]:0.3f} rad",
            )
            for line_index, line in enumerate(lines):
                draw.text((8, 5 + 20 * line_index), line, fill=(255, 255, 255, 255), font=font)
            images.append(image)
    finally:
        renderer.close()
    if not images:
        raise RuntimeError("validated trace produced no render frames")
    _publish_gif(images, output)
    gif_sha256 = sha256_file(output)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "gmt_g1_recorded_trace_visualization",
        "label": LABEL,
        "inputs": {
            "trace_path": manifest["trace"]["path"],
            "trace_sha256": manifest["trace"]["sha256"],
            "trace_manifest_path": manifest_path.name,
            "trace_manifest_sha256": manifest_sha256,
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "support_files": support_files,
        },
        "render": {
            "format": "gif",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": OUTPUT_FPS,
            "frames": len(images),
            "state_operation": "mujoco_forward_on_recorded_qpos_qvel",
            "policy_executed": False,
            "dynamics_rerun": False,
            "learning_performed": False,
            "success_label_emitted": False,
        },
        "runtime": {
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "pillow": PIL.__version__,
        },
        "output": {
            "path": output.name,
            "sha256": gif_sha256,
            "size": output.stat().st_size,
        },
    }
    receipt_sha256 = write_json_receipt(receipt_path, receipt)
    return {
        "gif_path": str(output),
        "gif_sha256": gif_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "frames": len(images),
        "label": LABEL,
    }


def main() -> int:
    args = _parser().parse_args()
    manifest, abi, arrays, support_files = load_validated_replay(
        trace_path=args.trace,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        upstream_root=args.upstream_root,
    )
    result = render_recorded_trace(
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        abi=abi,
        arrays=arrays,
        support_files=support_files,
        upstream_root=args.upstream_root,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
