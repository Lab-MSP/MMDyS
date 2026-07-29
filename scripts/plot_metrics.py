#!/usr/bin/env python3
"""
Plot evaluation metrics from a test_metrics_best.json file.

Produces a single figure with three panels:
  1. Sample-level confusion matrix (normalised by true class)
  2. Subject-level confusion matrix (normalised by true class)
  3. F1 by task group (horizontal bar chart)

Usage:
    python scripts/plot_metrics.py \
        --json /path/to/test_metrics_best.json \
        [--out  /path/to/output_stem]   # saves .png and .pdf; default = same dir as json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
TASK_GROUP_LABELS = {
    "syllable": "Syllable",
    "character": "Character",
    "word": "Word",
    "sentence": "Sentence",
}

# Colour palette (colourblind-friendly)
CMAP_CM   = "Blues"
BAR_COLOR = "#2979FF"
BAR_EDGE  = "#1565C0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_cm(cm: np.ndarray) -> np.ndarray:
    """Row-normalise confusion matrix (fraction of true-class samples)."""
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    return cm.astype(float) / row_sums


def _plot_confusion(ax, cm_raw: np.ndarray, labels: list[str], title: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cm_norm = _normalise_cm(cm_raw)
    n = len(labels)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=CMAP_CM, vmin=0, vmax=1)

    # Colour bar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Fraction of true class", fontsize=14)

    # Cell annotations
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val_norm = cm_norm[i, j]
            val_raw  = int(cm_raw[i, j])
            text_color = "white" if val_norm > thresh else "black"
            ax.text(
                j, i,
                f"{val_norm:.2f}\n({val_raw})",
                ha="center", va="center",
                fontsize=9 if n <= 4 else 7,
                color=text_color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    #ax.set_xlabel("Predicted", fontsize=12)
    #ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)


def _plot_task_bar(ax, f1_by_group: dict[str, float]) -> None:
    # Fixed display order
    order = ["syllable", "character", "word", "sentence"]
    keys   = [k for k in order if k in f1_by_group] + \
             [k for k in f1_by_group if k not in order]
    values = [f1_by_group[k] for k in keys]
    display_labels = [TASK_GROUP_LABELS.get(k, k.capitalize()) for k in keys]

    y_pos = np.arange(len(keys))
    bars = ax.barh(
        y_pos, values,
        color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.8,
        height=0.55,
    )

    # Value labels at end of each bar
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=11, color="#333333",
        )

    # Mean line
    mean_f1 = float(np.mean(values))
    ax.axvline(mean_f1, color="#E53935", linestyle="--", linewidth=1.4,
               label=f"Mean = {mean_f1:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=12)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("F1 by Task Group", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(0, min(1.05, max(values) + 0.08))
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot test metrics from JSON")
    p.add_argument(
        "--json", type=Path, required=False,
        default=Path("/data/user_data/kroseroj/AV_SPEECH/MMDys-Claude/outputs/"
                     "msdm_av_fusion_moe_v4_official/seed_17/test_metrics_best.json"),
        help="Path to test_metrics_best.json",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output stem or filename (saves .png and .pdf). "
             "If a plain name with no directory is given, saves next to the JSON file. "
             "Defaults to 'metrics_plot' in the same dir as --json.",
    )
    return p.parse_args()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    if not args.json.exists():
        print(f"[error] JSON not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.json.read_text(encoding="utf-8"))

    sample_cm  = np.array(data["sample_confusion_matrix"],  dtype=np.int64)
    subject_cm = np.array(data["subject_confusion_matrix"], dtype=np.int64)
    f1_group   = data["f1_by_task_group"]

    # Summary stats for figure title annotation
    f4   = data.get("f1_final_4cls", float("nan"))
    subj = data.get("f1_subject_macro_4cls", float("nan"))
    samp = data.get("f1_sample_macro_4cls", float("nan"))

    # Figure layout: [sample_cm | subject_cm | task_bar]
    fig = plt.figure(figsize=(18, 5.5))
    gs  = fig.add_gridspec(1, 3, wspace=0.38, left=0.06, right=0.97,
                           top=0.88, bottom=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    _plot_confusion(ax1, sample_cm,  SEVERITY_LABELS, "Sample-level Confusion Matrix")
    _plot_confusion(ax2, subject_cm, SEVERITY_LABELS, "Subject-level Confusion Matrix")
    _plot_task_bar(ax3, f1_group)

    # # Top annotation with key metrics
    # fig.text(
    #     0.5, 0.97,
    #     f"F1 Final = {f4:.4f}   |   F1 Subject = {subj:.4f}   |   F1 Sample = {samp:.4f}",
    #     ha="center", va="top", fontsize=12, color="#444444",
    #     style="italic",
    # )

    if args.out is None:
        out_stem = args.json.parent / "metrics_plot"
    elif args.out.parent == Path("."):
        # plain name like "my_figure" — save next to the JSON
        out_stem = args.json.parent / args.out.stem
    else:
        out_stem = args.out.with_suffix("")
    png_path = out_stem.with_suffix(".png")
    pdf_path = out_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {png_path}")
    print(f"Saved → {pdf_path}")


if __name__ == "__main__":
    main()
