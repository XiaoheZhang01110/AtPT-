from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "redrawn_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_cjk_font():
    candidates = [
        "Songti SC",
        "Heiti SC",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


FONT = pick_cjk_font()
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Arial", "Helvetica", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.0,
        "xtick.major.size": 2.4,
        "ytick.major.size": 0,
        "legend.frameon": False,
    }
)


data = pd.DataFrame(
    {
        "策略": ["NH", "HH", "FH", "BH", "MAPPO", "I-MAPPO"],
        "AWT": [289, 261, 215, 235, 204, 188],
        "AHT": [0, 51, 42, 24, 18, 13],
        "ATT": [1278, 1553, 1467, 1344, 1260, 1223],
        "AOD": [7.0, 1.7, 1.8, 2.0, 1.8, 1.6],
    }
)

metric_meta = [
    ("AWT", "平均等待时间 / s"),
    ("AHT", "平均驻站时间 / s"),
    ("ATT", "平均旅行时间 / s"),
    ("AOD", "车辆占用离散度"),
]

method_order = ["NH", "HH", "FH", "BH", "MAPPO", "I-MAPPO"]
data["策略"] = pd.Categorical(data["策略"], categories=method_order, ordered=True)
data = data.sort_values("策略")

base_color = "#C9D2DD"
rule_color = "#8EA1B5"
rl_color = "#6D93C2"
highlight_color = "#D85C4A"
edge_color = "#374151"

color_map = {
    "NH": "#D9DEE6",
    "HH": rule_color,
    "FH": rule_color,
    "BH": rule_color,
    "MAPPO": rl_color,
    "I-MAPPO": highlight_color,
}


def annotate_best(ax, x, y, label):
    ax.scatter([x], [y], s=22, color=highlight_color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.text(
        x,
        y + 0.28,
        label,
        ha="center",
        va="bottom",
        color=highlight_color,
        fontsize=7,
        fontweight="bold",
    )



def draw_metric_panel(ax, metric, ylabel):
    values = data[metric].to_numpy(dtype=float)
    labels = data["策略"].astype(str).to_list()
    x = np.arange(len(labels))
    colors = [color_map[label] for label in labels]

    ax.bar(x, values, width=0.62, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_ylabel(ylabel, fontsize=8.5, fontweight="bold", labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")

    if metric == "AHT":
        controlled = np.array(labels) != "NH"
        best_idx = int(np.where(controlled)[0][np.argmin(values[controlled])])
    else:
        best_idx = int(np.argmin(values))
    best_value = values[best_idx]
    if metric == "AOD":
        best_label = f"{best_value:.1f}"
    else:
        best_label = f"{best_value:.0f}"
    annotate_best(ax, x[best_idx], best_value, best_label)

    ymax = max(values) * 1.18 if max(values) > 0 else 1
    ax.set_ylim(0, ymax)

    if metric == "AHT":
        ax.axhline(30, color="#6B7280", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.text(5.45, 30, "30 s阈值", ha="right", va="bottom", fontsize=6.5, color="#6B7280")
        ax.text(x[0], ymax * 0.04, "无驻站", ha="center", va="bottom", fontsize=6.5, color="#6B7280")


fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
axes = axes.ravel()

for ax, (metric, ylabel) in zip(axes, metric_meta):
    draw_metric_panel(ax, metric, ylabel)

base = OUT_DIR / "fig_t34_metrics_comparison"
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

for metric, ylabel in metric_meta:
    fig_single, ax_single = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)
    draw_metric_panel(ax_single, metric, ylabel)
    single_base = OUT_DIR / f"fig_t34_{metric.lower()}"
    fig_single.savefig(single_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_single)

print(base)
