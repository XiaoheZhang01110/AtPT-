from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


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
        "font.size": 7.5,
        "figure.facecolor": "white",
    }
)


INK = "#24313D"
MUTED = "#6B7480"
LINE = "#52606D"
GRID = "#D8DEE6"
BLUE = "#4878A8"
BLUE_LIGHT = "#E5EEF8"
TEAL = "#4E8D83"
TEAL_LIGHT = "#E4F1EF"
AMBER = "#B87A32"
AMBER_LIGHT = "#F5EAD8"
ROSE = "#B9575D"
ROSE_LIGHT = "#F4DFE1"
GRAY_LIGHT = "#F5F7FA"


def rounded_box(ax, xy, w, h, label, face, edge, *, lw=0.9, fs=7.6, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.055",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        joinstyle="round",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        fontweight=weight,
        linespacing=1.23,
    )
    return patch


def arrow(ax, start, end, *, color=LINE, rad=0.0, lw=1.0, scale=9):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def small_label(ax, x, y, text, *, color=MUTED, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=6.8, color=color)


def draw_bus(ax, x, y, body="#4878A8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.46,
            0.22,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            linewidth=0.8,
            edgecolor=body,
            facecolor="white",
        )
    )
    ax.add_patch(Rectangle((x + 0.06, y + 0.10), 0.11, 0.07, facecolor=body, edgecolor="none", alpha=0.72))
    ax.add_patch(Rectangle((x + 0.20, y + 0.10), 0.11, 0.07, facecolor=body, edgecolor="none", alpha=0.72))
    ax.add_patch(Rectangle((x + 0.34, y + 0.10), 0.07, 0.07, facecolor=body, edgecolor="none", alpha=0.72))
    ax.add_patch(Circle((x + 0.12, y - 0.005), 0.035, facecolor=INK, edgecolor="none"))
    ax.add_patch(Circle((x + 0.35, y - 0.005), 0.035, facecolor=INK, edgecolor="none"))


def draw_route(ax):
    xs = [1.3, 2.6, 3.9, 5.2, 6.5, 7.8, 9.1, 10.4]
    y = 5.22
    ax.plot([xs[0], xs[-1]], [y, y], color="#A7B0BA", linewidth=1.15)
    for k, x in enumerate(xs):
        ax.add_patch(Circle((x, y), 0.075, facecolor="white", edgecolor="#8D98A5", linewidth=0.9))
        if k in (2, 4, 6):
            small_label(ax, x, y - 0.24, f"$s_{{{k+1}}}$")
    draw_bus(ax, 2.08, y + 0.18, BLUE)
    draw_bus(ax, 4.75, y + 0.18, TEAL)
    draw_bus(ax, 7.35, y + 0.18, ROSE)
    small_label(ax, 3.48, 5.83, "$h^-_{i,t}$", color=BLUE)
    small_label(ax, 6.10, 5.83, "$h^+_{i,t}$", color=ROSE)
    ax.annotate("", xy=(4.68, 5.73), xytext=(2.54, 5.73), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.85))
    ax.annotate("", xy=(7.35, 5.73), xytext=(5.24, 5.73), arrowprops=dict(arrowstyle="<->", color=ROSE, lw=0.85))


def draw_template_reference(ax):
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 2.1)
    ax.axis("off")
    rounded_box(ax, (0.35, 0.62), 2.0, 0.7, "交通运行环境", GRAY_LIGHT, "#B8C1CC")
    rounded_box(ax, (3.0, 0.62), 2.0, 0.7, "事件触发决策", BLUE_LIGHT, BLUE)
    rounded_box(ax, (5.65, 0.62), 2.0, 0.7, "控制动作执行", TEAL_LIGHT, TEAL)
    rounded_box(ax, (8.3, 0.62), 2.0, 0.7, "反馈与学习", ROSE_LIGHT, ROSE)
    arrow(ax, (2.35, 0.97), (3.0, 0.97), color=LINE)
    arrow(ax, (5.0, 0.97), (5.65, 0.97), color=LINE)
    arrow(ax, (7.65, 0.97), (8.3, 0.97), color=LINE)
    arrow(ax, (9.3, 0.61), (4.0, 0.61), color=ROSE, rad=-0.18, lw=0.9)
    ax.text(0.36, 1.72, "选用模板：交通系统场景 + 事件驱动控制 + 学习反馈闭环", fontsize=7.6, color=INK, weight="bold")
    ax.text(0.36, 1.43, "视觉处理：白底、低饱和色、直接标签、少量箭头、按机制分层", fontsize=6.8, color=MUTED)


def draw_workflow(ax):
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ax.add_patch(Rectangle((0.25, 4.65), 11.2, 1.55, facecolor="white", edgecolor=GRID, linewidth=0.8))
    ax.text(0.42, 6.0, "公交运行环境", ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK)
    draw_route(ax)

    rounded_box(ax, (0.55, 3.25), 1.55, 0.68, "线路参数\n客流到达率", GRAY_LIGHT, "#B8C1CC")
    rounded_box(ax, (0.55, 2.30), 1.55, 0.68, "车辆位置\n载客状态", GRAY_LIGHT, "#B8C1CC")
    arrow(ax, (1.32, 3.24), (1.32, 2.98), color="#9AA4AF", scale=8)
    arrow(ax, (2.10, 2.64), (2.75, 2.64), color="#9AA4AF", scale=8)

    main_y = 2.22
    boxes = [
        (2.75, main_y, 1.35, 0.84, "车辆到站\n触发事件", BLUE_LIGHT, BLUE, "bold"),
        (4.38, main_y, 1.35, 0.84, "上下客服务\n计算$\\omega_{i,j}$", AMBER_LIGHT, AMBER, "normal"),
        (6.01, main_y, 1.35, 0.84, "观测状态\n$h^- , h^+ , o/Z$", BLUE_LIGHT, BLUE, "normal"),
        (7.64, main_y, 1.35, 0.84, "策略网络\n输出$a_{i,t}$", ROSE_LIGHT, ROSE, "bold"),
        (9.27, main_y, 1.35, 0.84, "驻站$\\Delta d$\n离站运行", TEAL_LIGHT, TEAL, "bold"),
    ]
    for item in boxes:
        rounded_box(ax, item[:2], item[2], item[3], item[4], item[5], item[6], weight=item[7])
    for x0 in [4.10, 5.73, 7.36, 8.99]:
        arrow(ax, (x0, main_y + 0.42), (x0 + 0.28, main_y + 0.42), color=LINE, scale=8)

    small_label(ax, 3.43, 1.91, "事件驱动", color=BLUE)
    small_label(ax, 5.06, 1.91, "需求扰动", color=AMBER)
    small_label(ax, 6.69, 1.91, "局部可观测", color=BLUE)
    small_label(ax, 8.32, 1.91, "连续动作", color=ROSE)
    small_label(ax, 9.95, 1.91, "环境更新", color=TEAL)

    rounded_box(ax, (4.30, 0.76), 1.70, 0.66, "奖励计算\n$-\\omega_1CV^2-\\omega_2T^w$", TEAL_LIGHT, TEAL, fs=7.1)
    rounded_box(ax, (6.44, 0.76), 1.58, 0.66, "轨迹样本\n经验缓存", GRAY_LIGHT, "#AAB3BF", fs=7.3)
    rounded_box(ax, (8.45, 0.76), 1.65, 0.66, "MAPPO更新\n策略参数", ROSE_LIGHT, ROSE, fs=7.3)
    arrow(ax, (9.95, main_y), (5.15, 1.45), color=TEAL, rad=0.12, lw=0.95, scale=8)
    arrow(ax, (6.00, 1.09), (6.44, 1.09), color=LINE, scale=8)
    arrow(ax, (8.02, 1.09), (8.45, 1.09), color=LINE, scale=8)
    arrow(ax, (9.28, 1.45), (8.35, 2.22), color=ROSE, rad=0.16, lw=0.9, scale=8)

    path = MplPath(
        [(10.62, 2.62), (11.15, 2.62), (11.15, 4.88), (9.84, 4.88)],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO],
    )
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=TEAL, linewidth=0.9))
    arrow(ax, (9.84, 4.88), (9.52, 5.22), color=TEAL, rad=0.05, lw=0.9, scale=8)
    small_label(ax, 10.88, 3.74, "离站后进入下一站段", color=TEAL, ha="left")

    ax.plot([2.45, 10.90, 10.90, 2.45, 2.45], [0.45, 0.45, 3.48, 3.48, 0.45], color=GRID, linewidth=0.85)
    ax.text(2.55, 3.66, "强化学习驻站控制闭环", ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(
        2.55,
        0.22,
        "注：训练阶段利用奖励更新策略；应用阶段车辆到站后按当前策略直接给出驻站时间。",
        ha="left",
        va="center",
        fontsize=6.6,
        color=MUTED,
    )


def main():
    template_fig, template_ax = plt.subplots(figsize=(7.2, 1.35))
    draw_template_reference(template_ax)
    template_out = OUT_DIR / "fig_bus_holding_workflow_selected_template.pdf"
    template_fig.savefig(template_out, bbox_inches="tight")
    plt.close(template_fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    draw_workflow(ax)
    out = OUT_DIR / "fig_bus_holding_workflow.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(out)
    print(template_out)


if __name__ == "__main__":
    main()
