"""Show every paired outcome from the independently reproduced Study020 score."""

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
from matplotlib.lines import Line2D

SCORE_SHA256 = "0a58b19ceb71fce8ee8dc301ab5e59a9170e4f312363c900ec8fb49cc24bece0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-helper", type=Path, required=True)
    args = parser.parse_args()
    encoded = args.score.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != SCORE_SHA256:
        raise ValueError("figure requires the exact independently reproduced Study020 score")
    data = json.loads(encoded)
    pairs = data["paired"]["pairs"]
    episodes = {(row["noise_seed"], row["policy"]): row for row in data["episodes"]}
    if len(pairs) != 16 or len(episodes) != 32:
        raise ValueError("figure must include all 16 noise pairs")
    for suffix in (".png", ".pdf", ".csv", ".export.json"):
        if args.output.with_suffix(suffix).exists():
            raise ValueError("figure output already exists")
    helper = args.export_helper.resolve(strict=True)
    sys.path.insert(0, str(helper.parent))
    from figure_export import export_figure

    colors, markers = {"initial": "#0072B2", "final": "#D55E00"}, {"initial": "o", "final": "s"}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.2), sharey=True)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.82, wspace=0.13)
        for index, pair in enumerate(pairs):
            rows = [episodes[(pair["noise_seed"], policy)] for policy in ("initial", "final")]
            for ax, metric in zip(axes, ("raw_total_reward_sum", "duration_seconds"), strict=True):
                ax.plot(
                    [row[metric] for row in rows],
                    [index - 0.12, index + 0.12],
                    color="#888888",
                    lw=1,
                )
                for offset, row in zip((-0.12, 0.12), rows, strict=True):
                    policy = row["policy"]
                    ax.scatter(
                        row[metric],
                        index + offset,
                        s=58,
                        marker=markers[policy],
                        color=colors[policy],
                        zorder=3,
                    )
                    if row["fall"]:
                        ax.scatter(
                            row[metric],
                            index + offset,
                            s=90,
                            marker="x",
                            color="#111111",
                            linewidths=1.5,
                            zorder=4,
                        )
        axes[0].set(
            yticks=range(16),
            yticklabels=[str(pair["noise_seed"]) for pair in pairs],
            ylabel="Paired action-noise seed",
            xlabel="Raw episode return (reward units)",
            title="A  Return falls with shorter survival",
            xlim=(0, 4700),
            ylim=(15.7, -0.7),
        )
        axes[1].set(
            xlabel="Observed episode length (seconds)",
            title="B  Initial: 5 falls  |  Trained: 11 falls",
            xlim=(0, 21),
        )
        axes[1].set_xticks((0, 5, 10, 15, 20))
        for ax in axes:
            ax.grid(axis="x", color="#dddddd", linewidth=0.7)
            ax.set_axisbelow(True)
        legend = [
            Line2D(
                [], [], color=colors[policy], marker=markers[policy], linestyle="none", label=label
            )
            for policy, label in (
                ("initial", "Initial checkpoint"),
                ("final", "Trained checkpoint"),
            )
        ]
        legend.append(Line2D([], [], color="#111111", marker="x", linestyle="none", label="Fall"))
        fig.legend(
            handles=legend, loc="upper center", bbox_to_anchor=(0.56, 0.92), ncol=3, frameon=False
        )
        fig.suptitle(
            "Study 020: training worsened sampled survival in this test", fontsize=17, y=0.975
        )
        fig.text(
            0.12,
            0.045,
            "Same reset; one training seed; all 16 paired noise sequences. No smoothing, exclusions or confidence intervals.\n"
            "A fall shortens the return-observation window. Reaching 20 s is not task success: both checkpoints have 0 full-task passes.",
            fontsize=10,
            linespacing=1.6,
        )
        export_figure(
            fig,
            args.output,
            formats=("png", "pdf"),
            dpi=160,
            write_manifest=True,
            mkdir=True,
            provenance={
                "source_score": str(args.score.resolve()),
                "source_score_sha256": SCORE_SHA256,
                "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "export_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "transformations": [
                    "all 16 pairs in declared seed order",
                    "within-row vertical offsets only for legibility",
                    "no normalization, smoothing or exclusions",
                ],
                "uncertainty": "all paired observations shown; no inference across training seeds",
                "censoring": "falls shorten return and later-phase exposure; duration shown alongside return",
                "destination": "local development report; no journal compliance claim",
                "alt_text": "Paired plots show episode return and duration for all 16 noise seeds. The initial policy falls 5 times and the trained policy 11 times. Six pairs change from survival to a fall and none change in the other direction. Neither policy completes the full task.",
                "software_guidance": "Scientific Agent Skills (2026), https://doi.org/10.48550/arXiv.2609.00065; current v2 checked 2026-09-07",
            },
        )
        plt.close(fig)
    with args.output.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({"figure": str(args.output), "pairs": len(pairs)}))


if __name__ == "__main__":
    main()
