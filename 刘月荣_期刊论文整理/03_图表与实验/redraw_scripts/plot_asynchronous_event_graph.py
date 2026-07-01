from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


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
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
    }
)


BLUE = "#34A9D8"
GREEN = "#2AAE68"
YELLOW = "#F1D84B"
RED = "#D64D45"
GRAY = "#667085"
LINE = "#2F3A45"


fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
ax.set_xlim(0.0, 10.2)
ax.set_ylim(0.3, 7.65)
ax.set_xlabel("时间", fontsize=9)
ax.set_ylabel("站点", fontsize=9)

stations = np.arange(1, 8)
for s in stations:
    ax.hlines(s, 0.6, 9.75, color="#D8E7F1", linewidth=0.9, linestyles=(0, (2, 2)))

ax.set_yticks(stations)
ax.set_yticklabels([fr"$s_{i}$" for i in stations])
ax.set_xticks([6.15, 6.85])
ax.set_xticklabels([r"$t-\Delta_{i,t}$", r"$t$"])
ax.tick_params(axis="both", length=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

window = Rectangle((6.15, 0.55), 0.70, 6.95, facecolor=RED, edgecolor=RED, alpha=0.13, linewidth=1.0)
ax.add_patch(window)
ax.vlines([6.15, 6.85], 0.55, 7.5, colors=RED, linestyles=(0, (4, 3)), linewidth=0.9, alpha=0.6)
ax.text(6.08, 6.62, "邻居事件\n时间窗", ha="right", va="center", fontsize=7.5, color=RED, linespacing=1.2)

bus_offsets = [0.7, 2.1, 3.55, 5.0, 6.25, 7.45, 8.65]
for k, x0 in enumerate(bus_offsets):
    x = x0 + 0.43 * (stations - 1)
    y = stations
    ax.plot(x, y, color="#E6B422", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    if 1 <= k <= 4:
        ax.text(x[-1] + 0.02, y[-1] + 0.45, fr"$b_{{i{(-4 + k):+d}}}$".replace("+0", ""), fontsize=11, color="#111827")

current_events = np.array([[5.30, 1], [5.82, 2], [6.28, 3], [6.82, 4]])
downstream_events = np.array(
    [
        [0.86, 1],
        [1.38, 2],
        [1.92, 3],
        [2.45, 4],
        [2.98, 5],
        [3.52, 6],
        [4.05, 7],
        [3.12, 1],
        [3.65, 2],
        [4.18, 3],
        [4.72, 4],
        [5.25, 5],
        [5.78, 6],
        [6.32, 7],
    ]
)
upstream_events = np.array([[6.75, 1], [7.25, 2]])
target = (6.82, 4)

ax.scatter(downstream_events[:, 0], downstream_events[:, 1], s=150, facecolor=BLUE, edgecolor="black", linewidth=1.2, zorder=3)
ax.scatter(current_events[:, 0], current_events[:, 1], s=150, facecolor=YELLOW, edgecolor="black", linewidth=1.2, zorder=4)
ax.scatter(upstream_events[:, 0], upstream_events[:, 1], s=150, facecolor=GREEN, edgecolor="black", linewidth=1.2, zorder=4)
ax.scatter([target[0]], [target[1]], s=180, facecolor=YELLOW, edgecolor="black", linewidth=1.4, zorder=5)

for point in [(6.28, 3), (6.32, 7), (6.75, 1)]:
    arrow = FancyArrowPatch(
        point,
        target,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        color=RED,
        shrinkA=8,
        shrinkB=8,
        zorder=6,
    )
    ax.add_patch(arrow)

ax.annotate(
    r"当前决策事件 $(i,t)$",
    xy=target,
    xytext=(7.20, 4.28),
    ha="left",
    va="center",
    fontsize=8,
    color=LINE,
    arrowprops=dict(arrowstyle="-", color=LINE, linewidth=0.8),
)
ax.text(6.50, 0.78, "事件图注意力聚合", ha="center", va="center", fontsize=8, color=RED)

legend_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=YELLOW, markeredgecolor="black", markersize=7.5, label="自身到站事件"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="black", markersize=7.5, label="下游车辆事件"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="black", markersize=7.5, label="上游车辆事件"),
    Line2D([0], [0], color=RED, linewidth=1.3, label="注意力边"),
]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.52, -0.26),
    ncol=4,
    frameon=False,
    handlelength=1.7,
    columnspacing=1.6,
)

out = OUT_DIR / "fig_asynchronous_event_graph.pdf"
fig.savefig(out, bbox_inches="tight")
print(out)
