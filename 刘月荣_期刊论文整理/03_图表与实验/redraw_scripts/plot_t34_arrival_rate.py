from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTCollection


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "t34_arrival_rates.csv"
OUTPUT = ROOT / "redrawn_figures" / "fig_t34_arrival_rate.pdf"
PREVIEW = Path("/tmp/fig_t34_arrival_rate_preview.png")
SONGTI_REGULAR = Path("/tmp/SongtiSC-Regular.ttf")


def read_data() -> tuple[list[str], np.ndarray]:
    with DATA.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["stop"] for row in rows]
    values = np.array([float(row["arrival_rate_per_min"]) for row in rows])
    return labels, values


def main() -> None:
    if not SONGTI_REGULAR.exists():
        collection = TTCollection("/System/Library/Fonts/Supplemental/Songti.ttc")
        collection.fonts[6].save(SONGTI_REGULAR)
    songti = FontProperties(fname=SONGTI_REGULAR, style="normal", weight="normal", size=24)
    mpl.rcParams.update(
        {
            "font.family": "STSong",
            "mathtext.fontset": "stix",
            "font.size": 24,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    labels, values = read_data()
    x = np.arange(1, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(12.2, 6.1), constrained_layout=True)
    bar_color = "#F2C29F"
    highlight_color = "#D97A43"
    line_color = "#2C629B"
    top_indices = np.argsort(values)[-4:]
    bar_colors = np.full(len(values), bar_color, dtype=object)
    bar_colors[top_indices] = highlight_color

    ax.bar(
        x,
        values,
        width=0.68,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
        zorder=2,
    )
    ax.plot(
        x,
        values,
        color=line_color,
        linewidth=2.0,
        marker="o",
        markersize=4.6,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=1.2,
        zorder=3,
    )

    mean_value = float(values.mean())
    ax.axhline(mean_value, color="#B9C7D8", linewidth=1.1, linestyle=(0, (3, 2)), zorder=1)
    ax.text(
        39.6,
        mean_value + 0.025,
        "均值",
        ha="right",
        va="bottom",
        color="#5D6875",
        fontproperties=songti,
        fontsize=24,
    )
    peak = int(np.argmax(values))
    ax.annotate(
        "高需求站点",
        xy=(x[peak], values[peak]),
        xytext=(x[peak] + 3.0, values[peak] + 0.04),
        ha="left",
        va="center",
        color=highlight_color,
        fontproperties=songti,
        arrowprops={"arrowstyle": "-|>", "color": highlight_color, "lw": 1.0},
    )

    ax.set_xlabel("站点编号", fontproperties=songti)
    ax.set_ylabel(r"乘客到达率 $\lambda_j$（人/min）", fontproperties=songti)
    ax.set_ylim(0, 1.62)
    ax.set_yticks(np.arange(0, 1.61, 0.2))
    shown_ticks = np.arange(1, 40, 2)
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels([str(i) for i in shown_ticks])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=24)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(songti)
        label.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")
    ax.margins(x=0.01)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PREVIEW, dpi=180, bbox_inches="tight", pad_inches=0.04)
    print(OUTPUT)


if __name__ == "__main__":
    main()
