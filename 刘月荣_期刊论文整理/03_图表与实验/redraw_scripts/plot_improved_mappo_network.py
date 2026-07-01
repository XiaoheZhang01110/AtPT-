from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
    }
)


BLUE = "#4F78A7"
BLUE_LIGHT = "#DDE8F4"
RED = "#C85A4A"
RED_LIGHT = "#F6DFDB"
GREEN = "#5F8C6B"
GREEN_LIGHT = "#E0ECE3"
GRAY = "#5F6B7A"
GRAY_LIGHT = "#F4F6F8"
LINE = "#3E4A59"


def box(ax, xy, w, h, text, fc, ec, fs=8, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=0.9,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#1F2933",
        fontweight=weight,
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color=LINE, rad=0, lw=1.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

box(ax, (0.35, 4.55), 1.42, 0.70, "当前车辆状态\n$s_{i,t}$", BLUE_LIGHT, BLUE, fs=8.5, weight="bold")
box(ax, (0.35, 3.25), 1.42, 0.70, "邻居事件集合\n$\\mathcal{N}_{i,t}$", GREEN_LIGHT, GREEN, fs=8.5, weight="bold")
box(ax, (0.35, 1.15), 1.42, 0.70, "全局/邻近状态\n$(s_{i-1},s_i,s_{i+1})$", GRAY_LIGHT, "#AAB3BF", fs=8.1)

box(ax, (2.18, 4.55), 1.25, 0.70, "状态编码\nFC + Tanh", BLUE_LIGHT, BLUE)
box(ax, (2.18, 3.25), 1.25, 0.70, "GAT聚合\n事件影响", GREEN_LIGHT, GREEN)
box(ax, (2.18, 1.15), 1.25, 0.70, "价值编码\nFC + Tanh", GRAY_LIGHT, "#AAB3BF")

box(ax, (4.10, 3.85), 1.15, 0.82, "特征融合\n$[h_{i,t},m_{i,t}]$", "#EEF3F8", "#8EA1B5", fs=8.3, weight="bold")
box(ax, (4.10, 1.15), 1.15, 0.70, "Critic\n价值网络", GRAY_LIGHT, "#AAB3BF", fs=8.5, weight="bold")

box(ax, (5.92, 3.85), 1.24, 0.82, "Actor\n策略网络", RED_LIGHT, RED, fs=8.5, weight="bold")
box(ax, (5.92, 1.15), 1.24, 0.70, "状态价值\n$V(s)$", GRAY_LIGHT, "#AAB3BF", fs=8.5)

box(ax, (7.85, 4.35), 0.92, 0.58, "均值\n$\\mu$", RED_LIGHT, RED, fs=8.3)
box(ax, (7.85, 3.55), 0.92, 0.58, "方差\n$\\sigma^2$", RED_LIGHT, RED, fs=8.3)
box(ax, (8.95, 3.95), 0.72, 0.58, "驻站动作\n$a_{i,t}$", "#FAECE9", RED, fs=8.0, weight="bold")

box(ax, (7.85, 1.15), 1.82, 0.70, "MAPPO更新\n优势函数与剪切目标", "#FAECE9", RED, fs=8.0)

arrow(ax, (1.77, 4.90), (2.18, 4.90), BLUE)
arrow(ax, (1.77, 3.60), (2.18, 3.60), GREEN)
arrow(ax, (1.77, 1.50), (2.18, 1.50), GRAY)
arrow(ax, (3.43, 4.90), (4.10, 4.38), BLUE)
arrow(ax, (3.43, 3.60), (4.10, 4.10), GREEN)
arrow(ax, (5.25, 4.26), (5.92, 4.26), RED)
arrow(ax, (7.16, 4.26), (7.85, 4.64), RED)
arrow(ax, (7.16, 4.26), (7.85, 3.84), RED)
arrow(ax, (8.77, 4.64), (8.95, 4.24), RED)
arrow(ax, (8.77, 3.84), (8.95, 4.24), RED)

arrow(ax, (3.43, 1.50), (4.10, 1.50), GRAY)
arrow(ax, (5.25, 1.50), (5.92, 1.50), GRAY)
arrow(ax, (7.16, 1.50), (7.85, 1.50), GRAY)
arrow(ax, (8.76, 3.55), (8.76, 1.85), RED, rad=0.0, lw=0.9)
arrow(ax, (6.54, 1.85), (6.54, 3.82), GRAY, rad=0.0, lw=0.9)

ax.text(5.05, 5.55, "Actor分支：利用事件图注意力增强驻站决策", ha="center", va="center", fontsize=8.5, color=RED)
ax.text(4.85, 2.18, "Critic分支：估计状态价值并稳定策略更新", ha="center", va="center", fontsize=8.2, color=GRAY)
ax.text(8.78, 2.25, "训练反馈", ha="center", va="center", fontsize=7.5, color=RED)

out = OUT_DIR / "fig_improved_mappo_network.pdf"
fig.savefig(out, bbox_inches="tight")
print(out)
