from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_matplotlib_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "03_图表与实验" / "source_data" / "ring_trajectory_raw"
OUT_DIR = ROOT / "03_图表与实验" / "redrawn_figures"
FONT_PATH = (
    ROOT
    / "00_原始材料"
    / "2022级-刘月荣-论文及相关资料"
    / "学位论文-终稿"
    / "simsun.ttc"
)
STRATEGIES = [
    ("NH", "NH"),
    ("HH", "HH"),
    ("SH", "SH"),
    ("FH", "FH"),
    ("BH", "BH"),
    ("MAPPO", "MAPPO"),
    ("improved_MAPPO", "改进MAPPO"),
]
COLORS = ["#3B6FB6", "#D9792B", "#3E8E62", "#C94F46", "#8A62B0", "#B39B22"]
FONT = FontProperties(fname=str(FONT_PATH), weight="normal", size=26)


def configure_style() -> None:
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
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", length=4.2, width=0.8, direction="out", pad=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)
        label.set_fontweight("normal")
        label.set_fontstyle("normal")


def break_at_wrap(position: np.ndarray) -> np.ndarray:
    plotted = position.astype(float).copy()
    wrap_indices = np.flatnonzero(np.diff(plotted) < -6) + 1
    plotted[wrap_indices] = np.nan
    return plotted


def plot_strategy(data_name: str, output_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 7.1), constrained_layout=True)
    for bus_id in range(6):
        data = pd.read_csv(DATA_ROOT / data_name / f"bus_{bus_id}.csv")
        time = data["time_step"].to_numpy(dtype=float)
        station = break_at_wrap(data["station_position"].to_numpy(dtype=float))
        ax.plot(
            time,
            station,
            color=COLORS[bus_id],
            linewidth=1.35,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.set_xlim(0, 28_800)
    ax.set_ylim(1, 13)
    ax.set_xticks([0, 5000, 10_000, 15_000, 20_000, 25_000])
    ax.set_yticks(np.arange(1, 14))
    ax.set_xlabel("时间/s", fontproperties=FONT)
    ax.set_ylabel("公交站点", fontproperties=FONT)
    style_axis(ax)
    output = OUT_DIR / f"fig6_trajectory_{output_name}.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for data_name, output_name in STRATEGIES:
        print(plot_strategy(data_name, output_name))


if __name__ == "__main__":
    main()
