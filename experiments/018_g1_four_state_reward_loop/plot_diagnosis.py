"""Plot the sealed Study018 height traces; no simulator or policy execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PINS = {
    "course_run_manifest.json": "33b5e25ea8dfa7a8e3a4f520e7386edce067b920d75201e8ffb81581bd96f705",
    "zero_residual_frames.jsonl": "a8f0c4bbe35023898984e17f61fb6a7215149ee36a1a259e55bab0fcbf7fb23e",
    "final_policy_frames.jsonl": "38729578d079db6105bc3b494a4fb557b4313ae962f38cad30b064f0d0d73947",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-helper", type=Path, required=True)
    args = parser.parse_args()
    source = {}
    for name, expected in PINS.items():
        encoded = (args.run / name).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError(f"sealed Study018 input differs: {name}")
        source[name] = encoded
    if any(args.output.with_suffix(ext).exists() for ext in (".png", ".pdf", ".csv")):
        raise ValueError("figure output already exists")
    helper = args.export_helper.resolve(strict=True)
    sys.path.insert(0, str(helper.parent))
    from figure_export import export_figure

    records = []
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 6.6), sharex=True, sharey=True, layout="constrained"
        )
        for ax, label, title in zip(
            axes,
            ("zero_residual", "final_policy"),
            (
                "Zero residual: 56/74 posture-compliant samples",
                "Trained A: 47/71 posture-compliant samples",
            ),
            strict=True,
        ):
            rows = [json.loads(line) for line in source[f"{label}_frames.jsonl"].splitlines()]
            if len(rows) != 1000:
                raise ValueError("expected full 1000-step Study018 trace")
            times = np.arange(1, 1001) * 0.02
            actual = np.array([r["metrics"]["root_height_m"] for r in rows])
            reference = np.array([r["trajectory"]["current_reference"][0] for r in rows])
            inside = np.array([1.0 <= r["metrics"]["progress_m"] < 2.0 for r in rows])
            if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                raise ValueError("missing or nonfinite height cannot be silently plotted")
            indices = np.flatnonzero(inside)
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise ValueError("display expects the sealed single region visit")
            ax.axvspan(
                times[indices[0]] - 0.01,
                times[indices[-1]] + 0.01,
                facecolor="#eeeeee",
                edgecolor="#888888",
                hatch="//",
                linewidth=0.6,
                label="Physical crouching region",
            )
            ax.plot(times, actual, color="#222222", linewidth=1.8, label="Actual robot height")
            ax.plot(
                times,
                reference,
                color="#0072B2",
                linestyle="--",
                linewidth=1.6,
                label="Post-step reference target",
            )
            ax.axhline(
                0.60, color="#B34700", linestyle=":", linewidth=1.5, label="0.60 m posture ceiling"
            )
            ax.axhline(0.50, color="#666666", linestyle="-.", linewidth=1.0)
            ax.text(
                7.95,
                0.505,
                "Depth target: at least one region sample ≤0.50 m",
                ha="right",
                va="bottom",
                fontsize=9,
                color="#444444",
            )
            ax.set(title=title, ylabel="Root height (m)", xlim=(0, 8), ylim=(0.30, 1.05))
            ax.grid(axis="y", alpha=0.15)
            for i, row in enumerate(rows):
                records.append(
                    (
                        label,
                        i + 1,
                        times[i],
                        actual[i],
                        reference[i],
                        bool(inside[i]),
                        row["executed_mode"],
                        row["executed_phase_seconds"],
                    )
                )
        axes[-1].set_xlabel("Episode time (s) — first 8 seconds of each 20-second episode")
        axes[0].legend(loc="lower right", fontsize=9, ncol=2)
        fig.suptitle("The reference rises during the physical crouching region", fontsize=15)
        export_figure(
            fig,
            args.output,
            formats=("png", "pdf"),
            dpi=160,
            write_manifest=True,
            provenance={
                "raw_data": {str(args.run / name): digest for name, digest in PINS.items()},
                "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "export_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "transformations": [
                    "No smoothing or normalization",
                    "Display x range 0-8 s; CSV retains all 20 s",
                    "Region from measured progress in [1,2), shading extends half a 20 ms sample around each endpoint",
                ],
                "uncertainty": "None estimated: one seed, one deterministic episode per policy; temporally correlated samples",
                "missing_data": "All1000 rows present; nonfinite heights rejected",
                "destination": "Local development report, not journal submission",
                "alt_text": "Two height-versus-time panels. Actual robot height and the post-step target rise above the posture ceiling during the shaded physical region in both episodes. Training lowers compliance from56/74 to47/71; neither episode reaches the depth target in that region.",
                "software_guidance": "Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065; current v2 checked2026-09-07",
            },
        )
        plt.close(fig)
    with args.output.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "policy",
                "control_step",
                "time_s",
                "actual_height_m",
                "poststep_target_height_m",
                "inside_physical_region",
                "executed_mode",
                "executed_phase_s",
            )
        )
        writer.writerows(records)
    print(json.dumps({"output": str(args.output), "source_rows": len(records)}))


if __name__ == "__main__":
    main()
