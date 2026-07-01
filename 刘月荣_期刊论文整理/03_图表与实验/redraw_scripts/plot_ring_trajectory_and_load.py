from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_matplotlib_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "00_原始材料"
    / "2022级-刘月荣-论文及相关资料"
    / "学位论文-终稿"
    / "figures"
)
OUT_DIR = ROOT / "03_图表与实验" / "redrawn_figures"
DATA_DIR = ROOT / "03_图表与实验" / "source_data" / "fig6_ring_simulation"
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
    ("改进的MAPPO", "改进MAPPO"),
]

# Matplotlib's default categorical colors used by the original simulation plots.
TRAJECTORY_SOURCE_COLORS = np.array(
    [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ],
    dtype=float,
)
LOAD_SOURCE_COLORS = np.array(
    [
        (255, 0, 0),
        (255, 140, 0),
        (0, 128, 0),
        (153, 50, 204),
        (65, 105, 225),
        (191, 191, 0),
    ],
    dtype=float,
)
PLOT_COLORS = [
    "#3B6FB6",
    "#D9792B",
    "#3E8E62",
    "#C94F46",
    "#8A62B0",
    "#B39B22",
]

MAIN_SIZE = 26
LEGEND_SIZE = 24
SONG = FontProperties(fname=str(FONT_PATH), weight="normal", size=MAIN_SIZE)
LEGEND_FONT = FontProperties(
    fname=str(FONT_PATH), weight="normal", size=LEGEND_SIZE
)


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
            "axes.linewidth": 0.9,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", length=4.2, width=0.8, direction="out", pad=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SONG)
        label.set_fontweight("normal")
        label.set_fontstyle("normal")


def color_mask(rgb: np.ndarray, color: np.ndarray, tolerance: float = 48) -> np.ndarray:
    distance = np.sqrt(np.sum((rgb.astype(float) - color) ** 2, axis=2))
    return distance < tolerance


def clusters(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    values = np.sort(values)
    groups = np.split(values, np.where(np.diff(values) > 2)[0] + 1)
    return [float(np.median(group)) for group in groups if len(group) > 0]


def trace_color_paths(mask: np.ndarray) -> list[np.ndarray]:
    active: list[dict[str, object]] = []
    completed: list[np.ndarray] = []
    for x in range(mask.shape[1]):
        ys = clusters(np.flatnonzero(mask[:, x]))
        unmatched = set(range(len(ys)))

        for track in active:
            last_x, last_y = track["points"][-1]  # type: ignore[index]
            gap = x - last_x
            if gap > 14:
                continue
            candidates = [
                (abs(ys[i] - last_y), i)
                for i in unmatched
                if abs(ys[i] - last_y) <= 10 * gap + 4
            ]
            if candidates:
                _, idx = min(candidates)
                track["points"].append((x, ys[idx]))  # type: ignore[union-attr]
                track["miss"] = 0
                unmatched.remove(idx)
            else:
                track["miss"] = int(track["miss"]) + 1

        still_active: list[dict[str, object]] = []
        for track in active:
            if int(track["miss"]) > 13:
                points = np.asarray(track["points"], dtype=float)
                if len(points) >= 12:
                    completed.append(points)
            else:
                still_active.append(track)
        active = still_active

        for idx in unmatched:
            active.append({"points": [(x, ys[idx])], "miss": 0})

    for track in active:
        points = np.asarray(track["points"], dtype=float)
        if len(points) >= 12:
            completed.append(points)
    return completed


def dilate(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant")
        result = np.zeros_like(mask)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(points) >= 35:
            components.append(np.asarray(points, dtype=int))
    return components


def trajectory_paths(source: Path) -> list[tuple[np.ndarray, int]]:
    rgb = np.asarray(Image.open(source).convert("RGB"))
    # Plotting rectangle in the original 600 x 400 px exports.
    left, right, top, bottom = 75, 540, 21, 341
    crop = rgb[top : bottom + 1, left : right + 1]
    paths: list[tuple[np.ndarray, int]] = []
    source_masks = [color_mask(crop, color, tolerance=62) for color in TRAJECTORY_SOURCE_COLORS]
    union_mask = np.logical_or.reduce(source_masks)
    for component_index, component in enumerate(connected_components(dilate(union_mask, 1))):
        ys, xs = component[:, 0], component[:, 1]
        unique_x = np.unique(xs)
        if len(unique_x) < 10:
            continue
        center_y = np.array([np.median(ys[xs == x]) for x in unique_x])
        x_data = unique_x / (right - left) * 28800
        station = 13 - center_y / (bottom - top) * 12
        nearest_stop = np.rint(station)
        station = np.where(np.abs(station - nearest_stop) <= 0.16, nearest_stop, station)
        # Within one circuit, a bus can only move forward or remain at a stop.
        # Downward steps are tracing artefacts caused by overlapping raster lines.
        station = np.maximum.accumulate(station)
        station = np.clip(station, 1, 13)
        paths.append((np.column_stack([x_data, station]), component_index))
    return paths


def find_plot_bounds(rgb: np.ndarray) -> tuple[int, int, int, int]:
    dark = np.all(rgb < 70, axis=2)
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    horizontal = np.flatnonzero(row_counts > rgb.shape[1] * 0.55)
    vertical = np.flatnonzero(col_counts > rgb.shape[0] * 0.55)
    if len(horizontal) < 2 or len(vertical) < 2:
        raise RuntimeError("Could not identify the plotting rectangle")
    return int(vertical[0]), int(vertical[-1]), int(horizontal[0]), int(horizontal[-1])


def y_calibration(rgb: np.ndarray, left: int, top: int, bottom: int) -> tuple[float, float]:
    dark = np.all(rgb < 90, axis=2)
    # Tick marks extend into the plotting area from the left spine.
    strip = dark[top : bottom + 1, left + 1 : left + 9]
    counts = strip.sum(axis=1)
    candidates = np.flatnonzero(counts >= 4) + top
    groups = np.split(candidates, np.where(np.diff(candidates) > 2)[0] + 1)
    tick_y = np.array(
        [np.median(group) for group in groups if len(group) and top + 5 < np.median(group) < bottom - 5]
    )
    if len(tick_y) < 3:
        raise RuntimeError("Could not identify y-axis ticks")
    tick_y = np.sort(tick_y)[::-1]
    pixel_step = float(np.median(np.diff(tick_y)))
    return float(tick_y[0]), pixel_step


def load_data(source: Path, strategy: str) -> pd.DataFrame:
    rgb = np.asarray(Image.open(source).convert("RGB"))
    left, right, top, bottom = find_plot_bounds(rgb)
    tick_bottom, pixel_step = y_calibration(rgb, left, top, bottom)

    # Lowest labelled tick in each original panel.
    tick_start = {
        "NH": 300,
        "HH": 300,
        "SH": 300,
        "FH": 300,
        "BH": 300,
        "MAPPO": 400,
        "改进MAPPO": 400,
    }[strategy]
    x_positions = left + (np.arange(1, 13) - 0.5) / 12 * (right - left)
    values: dict[str, list[float]] = {"站点": list(range(1, 13))}
    recovered: list[list[float]] = []
    upper_left_legend = strategy in {"FH", "BH", "MAPPO", "改进MAPPO"}

    for bus_index, source_color in enumerate(LOAD_SOURCE_COLORS, start=1):
        mask = color_mask(rgb, source_color, tolerance=32)
        if upper_left_legend:
            mask[top : min(bottom + 1, top + 155), max(0, left - 12) : left + 78] = False
        else:
            mask[max(top, bottom - 155) : bottom + 1, right - 115 : right + 1] = False
        bus_values: list[float] = []
        for x in x_positions:
            x0, x1 = max(0, int(round(x)) - 4), min(rgb.shape[1], int(round(x)) + 5)
            ys, _ = np.where(mask[:, x0:x1])
            ys = ys[(ys >= top) & (ys <= bottom)]
            if len(ys) == 0:
                bus_values.append(np.nan)
                continue
            y_pixel = float(np.median(ys))
            value = tick_start + (y_pixel - tick_bottom) / pixel_step * 100
            bus_values.append(float(np.round(value)))
        recovered.append(bus_values)

    matrix = np.asarray(recovered, dtype=float)
    # At overlapping curves and beneath the old legend, exact source-color pixels
    # can be absent. The same-station fleet median preserves the observed profile
    # better than interpolating across several stations.
    station_medians = np.nanmedian(matrix, axis=0)
    for row in range(matrix.shape[0]):
        missing = np.isnan(matrix[row])
        matrix[row, missing] = station_medians[missing]
        series = pd.Series(matrix[row], dtype=float).interpolate(limit_direction="both")
        values[f"车辆{row + 1}"] = series.round().tolist()
    return pd.DataFrame(values)


def plot_trajectory(strategy_file: str, strategy_label: str) -> Path:
    paths = trajectory_paths(SOURCE_DIR / f"{strategy_file}车辆轨迹.png")
    fig, ax = plt.subplots(figsize=(10.2, 7.1), constrained_layout=True)
    for path, color_index in paths:
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=PLOT_COLORS[color_index % len(PLOT_COLORS)],
            linewidth=1.45,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.set_xlim(0, 28800)
    ax.set_ylim(1, 13)
    ax.set_xticks(np.arange(0, 28801, 5000))
    ax.set_yticks(np.arange(1, 14))
    ax.set_xlabel("时间/s", fontproperties=SONG)
    ax.set_ylabel("公交站点", fontproperties=SONG)
    style_axis(ax)
    out = OUT_DIR / f"fig6_trajectory_{strategy_label}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_load(strategy_file: str, strategy_label: str) -> Path:
    data = load_data(SOURCE_DIR / f"{strategy_file}载客量.png", strategy_label)
    data.to_csv(DATA_DIR / f"load_{strategy_label}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.2, 7.1), constrained_layout=True)
    for index in range(1, 7):
        ax.plot(
            data["站点"],
            data[f"车辆{index}"],
            color=PLOT_COLORS[index - 1],
            linewidth=2.15,
            label=f"车辆{index}",
        )
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel("公交站点", fontproperties=SONG)
    ax.set_ylabel("载客量/人", fontproperties=SONG)
    style_axis(ax)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        prop=LEGEND_FONT,
        handlelength=1.6,
        handletextpad=0.5,
        borderaxespad=0.3,
        labelspacing=0.25,
        ncol=1,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")
        text.set_fontstyle("normal")
    out = OUT_DIR / f"fig6_load_{strategy_label}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for strategy_file, strategy_label in STRATEGIES:
        outputs.append(plot_trajectory(strategy_file, strategy_label))
        outputs.append(plot_load(strategy_file, strategy_label))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
