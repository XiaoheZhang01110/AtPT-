from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/codex_matplotlib_cache",
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
DATA = (
    ROOT
    / "00_原始材料"
    / "2022级-刘月荣-论文及相关资料"
    / "代码"
    / "a_biyelunwen2_15"
    / "base_amappo"
    / "log"
    / "A_0_1mappo_caac_2_0_21.csv"
)
OUT_DIR = ROOT / "03_图表与实验" / "redrawn_figures"
OUT = OUT_DIR / "fig5_mappo_training_curves.pdf"
SEPARATE_OUT = {
    "reward": OUT_DIR / "fig5a_mappo_cumulative_reward.pdf",
    "actor": OUT_DIR / "fig5b_mappo_actor_loss.pdf",
    "critic": OUT_DIR / "fig5c_mappo_critic_mse.pdf",
}

FONT_PATH = (
    ROOT
    / "00_原始材料"
    / "2022级-刘月荣-论文及相关资料"
    / "学位论文-终稿"
    / "simsun.ttc"
)
MAIN_FONT_SIZE = 26
LEGEND_FONT_SIZE = 24
SONG = FontProperties(fname=str(FONT_PATH), weight="normal", size=MAIN_FONT_SIZE)
LEGEND_FONT = FontProperties(fname=str(FONT_PATH), weight="normal", size=LEGEND_FONT_SIZE)


def moving_average(values: pd.Series, window: int = 10) -> pd.Series:
    return values.rolling(window=window, min_periods=window).mean()


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.55)
    ax.grid(axis="x", color="#EEF1F4", linewidth=0.35)
    ax.tick_params(axis="both", length=3.2, width=0.7, pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SONG)
        label.set_fontsize(MAIN_FONT_SIZE)
        label.set_fontweight("normal")
        label.set_fontstyle("normal")


def enforce_axis_font(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontproperties(SONG)
    ax.xaxis.label.set_fontsize(MAIN_FONT_SIZE)
    ax.xaxis.label.set_fontweight("normal")
    ax.xaxis.label.set_fontstyle("normal")
    ax.yaxis.label.set_fontproperties(SONG)
    ax.yaxis.label.set_fontsize(MAIN_FONT_SIZE)
    ax.yaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontstyle("normal")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SONG)
        label.set_fontsize(MAIN_FONT_SIZE)
        label.set_fontweight("normal")
        label.set_fontstyle("normal")


def plot_series(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    ylabel: str,
    line_color: str,
    smooth_color: str,
    ypad: float = 0.08,
    raw_alpha: float = 0.48,
    raw_lw: float = 0.55,
    label_fontsize: float = MAIN_FONT_SIZE,
) -> None:
    ax.plot(x, y, color=smooth_color, linewidth=raw_lw, alpha=raw_alpha, label="原始值")
    ax.plot(x, moving_average(y), color=smooth_color, linewidth=1.75, label="10轮滑动平均")
    ax.set_ylabel(ylabel, fontproperties=SONG, fontsize=label_fontsize, fontweight="normal")
    ax.yaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontstyle("normal")
    style_axis(ax)
    ymin, ymax = float(y.min()), float(y.max())
    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
    ax.set_ylim(ymin - span * ypad, ymax + span * ypad)


def add_curve_legend(ax: plt.Axes, loc: str = "lower right") -> None:
    legend = ax.legend(
        loc=loc,
        frameon=False,
        prop=LEGEND_FONT,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    for line in legend.get_lines():
        line.set_linewidth(1.2)
    for text in legend.get_texts():
        text.set_fontproperties(LEGEND_FONT)
        text.set_fontweight("normal")
        text.set_fontstyle("normal")


def main() -> None:
    df = pd.read_csv(DATA)
    episode = pd.Series(range(1, len(df) + 1), name="episode")

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
        }
    )

    panel_specs = [
        (
            "reward",
            "累计奖励",
            df["reward"],
            "#AFCBEA",
            "#2F6FA7",
            None,
            0.08,
            0.48,
            0.55,
            MAIN_FONT_SIZE,
        ),
        (
            "actor",
            "Actor损失",
            df["ploss"],
            "#E8B6A5",
            "#B65B45",
            0,
            0.08,
            0.48,
            0.55,
            MAIN_FONT_SIZE,
        ),
        (
            "critic",
            "Critic均方误差",
            df["qloss"],
            "#BFD7B5",
            "#4F8A5B",
            None,
            0.04,
            0.72,
            0.65,
            MAIN_FONT_SIZE,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    for ax, (_, ylabel, y, line_color, smooth_color, hline, ypad, raw_alpha, raw_lw, label_fontsize) in zip(axes, panel_specs):
        plot_series(
            ax,
            episode,
            y,
            ylabel,
            line_color,
            smooth_color,
            ypad=ypad,
            raw_alpha=raw_alpha,
            raw_lw=raw_lw,
            label_fontsize=label_fontsize,
        )
        if hline is not None:
            ax.axhline(hline, color="#555555", linewidth=0.55, alpha=0.75)

    for ax in axes:
        ax.set_xlabel("训练轮次", fontproperties=SONG, fontsize=MAIN_FONT_SIZE)
        ax.xaxis.label.set_fontweight("normal")
        ax.xaxis.label.set_fontstyle("normal")
        ax.set_xlim(0, len(df))
        ax.set_xticks([0, 100, 200, 300])
        enforce_axis_font(ax)

    add_curve_legend(axes[0], loc="lower right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)

    for key, ylabel, y, line_color, smooth_color, hline, ypad, raw_alpha, raw_lw, label_fontsize in panel_specs:
        fig_single, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        plot_series(
            ax,
            episode,
            y,
            ylabel,
            line_color,
            smooth_color,
            ypad=ypad,
            raw_alpha=raw_alpha,
            raw_lw=raw_lw,
            label_fontsize=label_fontsize,
        )
        if hline is not None:
            ax.axhline(hline, color="#555555", linewidth=0.55, alpha=0.75)
        ax.set_xlabel("训练轮次", fontproperties=SONG, fontsize=MAIN_FONT_SIZE, fontweight="normal")
        ax.xaxis.label.set_fontweight("normal")
        ax.xaxis.label.set_fontstyle("normal")
        ax.set_xlim(0, len(df))
        ax.set_xticks([0, 100, 200, 300])
        enforce_axis_font(ax)
        legend_locs = {
            "reward": "lower right",
            "actor": "upper left",
            "critic": "upper right",
        }
        add_curve_legend(ax, loc=legend_locs[key])
        fig_single.savefig(SEPARATE_OUT[key], bbox_inches="tight")
        plt.close(fig_single)

    print(OUT)
    for path in SEPARATE_OUT.values():
        print(path)


if __name__ == "__main__":
    main()
