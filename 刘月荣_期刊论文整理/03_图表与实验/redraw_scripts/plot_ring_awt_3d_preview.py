import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "source_data" / "ring_metrics" / "awt_by_stop_3d_preview.csv"
OUT_DIR = ROOT / "redrawn_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_CJK = "Songti SC"
FONT_LATIN = "Times New Roman"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [FONT_LATIN, FONT_CJK],
        "font.size": 26,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

METHODS = ["NH", "HH", "SH", "FH", "BH", "MAPPO", "improved_MAPPO"]
DISPLAY_NAMES = {
    "NH": "NH",
    "HH": "HH",
    "SH": "SH",
    "FH": "FH",
    "BH": "BH",
    "MAPPO": "MAPPO",
    "improved_MAPPO": "改进MAPPO",
}
COLORS = {
    "NH": "#B8BEC7",
    "HH": "#5B8DB8",
    "SH": "#7BAA8B",
    "FH": "#D6A85F",
    "BH": "#8B79A8",
    "MAPPO": "#D9785F",
    "improved_MAPPO": "#B33B45",
}


def draw() -> tuple[plt.Figure, plt.Axes]:
    data = pd.read_csv(DATA_PATH)
    x = np.arange(1, len(data) + 1)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    dx = 0.58
    dy = 0.56
    for method_index, method in enumerate(METHODS):
        values = data[method].to_numpy(dtype=float)
        y = np.full_like(x, method_index, dtype=float)
        ax.bar3d(
            x - dx / 2,
            y - dy / 2,
            np.zeros_like(values),
            dx,
            dy,
            values,
            color=COLORS[method],
            edgecolor="#4B5563",
            linewidth=0.35,
            shade=True,
            alpha=0.96,
            zsort="average",
        )

    ax.set_xlabel("公交站点", labelpad=28, fontfamily=FONT_CJK)
    ax.set_ylabel("控制策略", labelpad=34, fontfamily=FONT_CJK)
    ax.set_zlabel("")
    fig.text(
        0.965,
        0.53,
        "平均等待时间 / s",
        rotation=90,
        ha="center",
        va="center",
        fontsize=26,
        fontfamily=FONT_CJK,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([rf"$s_{{{i}}}$" for i in x], fontfamily=FONT_LATIN)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels(
        [str(i) for i in range(1, len(METHODS) + 1)],
        fontfamily=FONT_LATIN,
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="x", pad=8, labelsize=22)
    ax.tick_params(axis="y", pad=4, labelsize=22)
    ax.tick_params(axis="z", pad=8, labelsize=22)

    ax.set_xlim(0.35, 12.75)
    ax.set_ylim(-0.6, len(METHODS) - 0.1)
    ax.set_zlim(0, 1800)
    ax.set_zticks(np.arange(0, 1801, 300))
    ax.view_init(elev=25, azim=-56)
    ax.set_box_aspect((2.15, 1.12, 1.0))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis.pane.set_edgecolor("#D1D5DB")
        axis._axinfo["grid"]["color"] = (0.83, 0.85, 0.88, 0.75)
        axis._axinfo["grid"]["linewidth"] = 0.7

    legend_handles = [
        Patch(facecolor=COLORS[m], edgecolor="#4B5563", linewidth=0.35, label=DISPLAY_NAMES[m])
        for m in METHODS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=4,
        frameon=False,
        prop={"family": FONT_CJK, "size": 20},
        handlelength=1.2,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.04, top=0.99)
    return fig, ax


if __name__ == "__main__":
    figure, _ = draw()
    output = OUT_DIR / "fig_ring_awt_3d_preview"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(output)
