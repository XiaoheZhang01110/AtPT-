import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "source_data" / "ring_metrics" / "ring_metrics_by_stop.csv"
OUT_DIR = ROOT / "redrawn_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_CJK = "Songti SC"
FONT_LATIN = "Times New Roman"
SONGTI_REGULAR = Path("/private/tmp/SongtiSC-Regular.ttf")
if not SONGTI_REGULAR.exists():
    TTFont("/System/Library/Fonts/Supplemental/Songti.ttc", fontNumber=6).save(
        SONGTI_REGULAR
    )
font_manager.fontManager.addfont(SONGTI_REGULAR)
CJK_PROP = font_manager.FontProperties(fname=SONGTI_REGULAR, weight="normal")

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [FONT_LATIN, FONT_CJK],
        "font.size": 26,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.linewidth": 1.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.unicode_minus": False,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
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
STYLES = {
    "NH": dict(color="#9AA1AA", marker="o", linestyle=(0, (5, 3)), linewidth=2.2, markersize=8),
    "HH": dict(color="#4F83B3", marker="s", linestyle="-", linewidth=2.2, markersize=8),
    "SH": dict(color="#6FA37F", marker="^", linestyle="-", linewidth=2.2, markersize=8),
    "FH": dict(color="#D2A04F", marker="D", linestyle="-", linewidth=2.2, markersize=8),
    "BH": dict(color="#7B6A9F", marker="v", linestyle="-", linewidth=2.2, markersize=8),
    "MAPPO": dict(color="#D36B50", marker="P", linestyle="-", linewidth=2.8, markersize=9),
    "improved_MAPPO": dict(
        color="#A92835", marker="*", linestyle="-", linewidth=3.2, markersize=13
    ),
}
YLABELS = {
    "AWT": "平均等待时间 / s",
    "AHT": "平均驻站时间 / s",
    "ATT": "平均旅行时间 / s",
    "AOD": "平均载客量离散度",
}


def draw_metric(data: pd.DataFrame, metric: str) -> plt.Figure:
    subset = data[data["metric"] == metric]
    fig, ax = plt.subplots(figsize=(12.8, 7.6))

    for method in METHODS:
        method_data = subset[subset["method"] == method].sort_values("stop")
        ax.plot(
            method_data["stop"],
            method_data["value"],
            label=DISPLAY_NAMES[method],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=4 if method in {"MAPPO", "improved_MAPPO"} else 3,
            **STYLES[method],
        )

    stops = np.arange(1, 13)
    ax.set_xlim(0.65, 12.35)
    ax.set_xticks(stops)
    stop_labels = [rf"$\mathrm{{s}}_{{{stop}}}$" for stop in stops]
    ax.set_xticklabels(
        stop_labels,
        fontfamily=FONT_LATIN,
        fontweight="normal",
        fontstyle="normal",
    )
    ax.set_xlabel("公交站点", fontsize=26, labelpad=12, fontproperties=CJK_PROP)
    ax.set_ylabel(YLABELS[metric], fontsize=26, labelpad=14, fontproperties=CJK_PROP)
    ax.tick_params(axis="both", labelsize=26)

    ymax = float(subset["value"].max())
    ax.set_ylim(0, ymax * 1.16 if ymax > 0 else 1)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.9, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        prop=CJK_PROP.copy(),
        handlelength=2.1,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    for text in ax.get_legend().get_texts():
        text.set_fontsize(24)
        text.set_fontweight("normal")

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.80)
    return fig


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    for metric in YLABELS:
        figure = draw_metric(data, metric)
        output = OUT_DIR / f"fig_ring_{metric.lower()}_by_stop"
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(figure)
        print(output)


if __name__ == "__main__":
    main()
