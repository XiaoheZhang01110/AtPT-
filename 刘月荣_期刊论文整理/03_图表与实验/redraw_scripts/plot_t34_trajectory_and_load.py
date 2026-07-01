from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_matplotlib_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ROOT / "00_原始材料" / "2022级-刘月荣-论文及相关资料"
FIGURE_DIR = MATERIALS / "学位论文-终稿" / "figures"
CODE_DIR = MATERIALS / "代码" / "a_biyelunwen2_15"
OUT_DIR = ROOT / "03_图表与实验" / "redrawn_figures"
DATA_DIR = ROOT / "03_图表与实验" / "source_data" / "t34_trajectory_load"
PREVIEW_DIR = Path("/tmp/t34_trajectory_load_previews")
FONT_PATH = MATERIALS / "学位论文-终稿" / "simsun.ttc"

STRATEGIES = [
    ("NH", "line_NH", None),
    ("HH", "line_HH", None),
    ("FH", "line_FH", None),
    ("BH", "line_BH", None),
    ("MAPPO", "line_mappo", CODE_DIR / "line_mappo" / "log" / "trajectory"),
    ("改进MAPPO", "line_改进的mappo", CODE_DIR / "line_amappo" / "log" / "trajectory"),
]

PLOT_COLORS = [
    "#3B6FB6",
    "#D9792B",
    "#3E8E62",
    "#C94F46",
    "#8A62B0",
    "#8B6A52",
    "#C66A9B",
    "#707070",
    "#A49A24",
    "#2694A6",
]
MAIN_SIZE = 26
SONG = FontProperties(fname=str(FONT_PATH), weight="normal", size=MAIN_SIZE)


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


def find_plot_bounds(rgb: np.ndarray) -> tuple[int, int, int, int]:
    dark = np.all(rgb < 70, axis=2)
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    horizontal = np.flatnonzero(row_counts > rgb.shape[1] * 0.55)
    vertical = np.flatnonzero(col_counts > rgb.shape[0] * 0.55)
    if len(horizontal) < 2 or len(vertical) < 2:
        raise RuntimeError("Could not identify the plotting rectangle")
    return int(vertical[0]), int(vertical[-1]), int(horizontal[0]), int(horizontal[-1])


def source_colors(rgb: np.ndarray, limit: int = 30) -> list[np.ndarray]:
    pixels, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    colors: list[np.ndarray] = []
    for index in order:
        color = pixels[index].astype(float)
        if counts[index] < 35:
            break
        if color.max() - color.min() < 45 or color.min() > 230:
            continue
        if any(np.linalg.norm(color - selected) < 28 for selected in colors):
            continue
        colors.append(color)
        if len(colors) == limit:
            break
    return colors


def color_mask(rgb: np.ndarray, color: np.ndarray, tolerance: float = 42) -> np.ndarray:
    distance = np.sqrt(np.sum((rgb.astype(float) - color) ** 2, axis=2))
    return distance < tolerance


def clusters(values: np.ndarray, gap: int = 3) -> list[float]:
    if values.size == 0:
        return []
    values = np.sort(values)
    groups = np.split(values, np.where(np.diff(values) > gap)[0] + 1)
    return [float(np.median(group)) for group in groups if len(group)]


def trace_color_paths(mask: np.ndarray) -> list[np.ndarray]:
    active: list[dict[str, object]] = []
    completed: list[np.ndarray] = []
    for x in range(mask.shape[1]):
        ys = clusters(np.flatnonzero(mask[:, x]), gap=3)
        unmatched = set(range(len(ys)))
        for track in active:
            last_x, last_y = track["points"][-1]  # type: ignore[index]
            gap = x - last_x
            if gap > 10:
                continue
            candidates = [
                (abs(ys[i] - last_y), i)
                for i in unmatched
                if abs(ys[i] - last_y) <= 5 * gap + 3
            ]
            if candidates:
                _, index = min(candidates)
                track["points"].append((x, ys[index]))  # type: ignore[union-attr]
                track["miss"] = 0
                unmatched.remove(index)
            else:
                track["miss"] = int(track["miss"]) + 1
        still_active: list[dict[str, object]] = []
        for track in active:
            if int(track["miss"]) > 9:
                points = np.asarray(track["points"], dtype=float)
                if len(points) >= 25:
                    completed.append(points)
            else:
                still_active.append(track)
        active = still_active
        for index in unmatched:
            active.append({"points": [(x, ys[index])], "miss": 0})
    for track in active:
        points = np.asarray(track["points"], dtype=float)
        if len(points) >= 25:
            completed.append(points)
    return completed


def digitize_trajectory(source: Path) -> list[tuple[np.ndarray, int]]:
    rgb = np.asarray(Image.open(source).convert("RGB"))
    left, right, top, bottom = find_plot_bounds(rgb)
    crop = rgb[top : bottom + 1, left : right + 1]
    paths: list[tuple[np.ndarray, int]] = []
    for color_index, color in enumerate(source_colors(crop, limit=12)):
        for pixels in trace_color_paths(color_mask(crop, color)):
            if np.ptp(pixels[:, 0]) < crop.shape[1] * 0.15:
                continue
            time = pixels[:, 0] / (right - left) * 12000
            station = 39 - pixels[:, 1] / (bottom - top) * 38
            station = np.maximum.accumulate(station)
            station = np.clip(station, 1, 39)
            paths.append((np.column_stack([time, station]), color_index))
    return sorted(paths, key=lambda item: item[0][0, 0])


def read_raw_buses(log_dir: Path) -> list[dict[str, np.ndarray]]:
    buses: list[dict[str, np.ndarray]] = []
    for path in sorted(log_dir.glob("*.csv"), key=lambda p: int(p.stem)):
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        buses.append(
            {
                "time": np.array([float(row["time"]) - 900 for row in rows]),
                "loc": np.array([float(row["loc"]) + 1 for row in rows]),
                "load": np.array([float(row["op"]) * 80 for row in rows]),
            }
        )
    return buses


def raw_load_by_station(buses: list[dict[str, np.ndarray]]) -> np.ndarray:
    matrix = np.full((len(buses), 39), np.nan)
    for bus_index, bus in enumerate(buses):
        for station in range(1, 40):
            near_stop = np.abs(bus["loc"] - station) <= 0.035
            if np.any(near_stop):
                matrix[bus_index, station - 1] = np.max(bus["load"][near_stop])
        valid = np.flatnonzero(~np.isnan(matrix[bus_index]))
        if len(valid):
            matrix[bus_index] = np.interp(np.arange(39), valid, matrix[bus_index, valid])
    return matrix


def digitize_load(source: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(source).convert("RGB"))
    left, right, top, bottom = find_plot_bounds(rgb)
    crop = rgb[top : bottom + 1, left : right + 1]
    x_positions = (np.arange(1, 40) - 1) / 38 * (right - left)
    tracks: list[np.ndarray] = []
    for color in source_colors(crop, limit=30):
        mask = color_mask(crop, color, tolerance=38)
        station_clusters: list[list[float]] = []
        for x in x_positions:
            x0 = max(0, int(round(x)) - 3)
            x1 = min(crop.shape[1], int(round(x)) + 4)
            ys, _ = np.where(mask[:, x0:x1])
            station_clusters.append(clusters(ys, gap=4))
        track_count = max((len(group) for group in station_clusters), default=0)
        for rank in range(track_count):
            values = np.full(39, np.nan)
            for station, group in enumerate(station_clusters):
                ordered = sorted(group)
                if rank < len(ordered):
                    values[station] = (bottom - top - ordered[rank]) / (bottom - top) * 70
            valid = np.flatnonzero(~np.isnan(values))
            if len(valid) >= 30:
                values = np.interp(np.arange(39), valid, values[valid])
                values[[0, -1]] = 0
                tracks.append(values)
    if not tracks:
        raise RuntimeError(f"No load curves recovered from {source}")
    # Keep the 25 most complete visible curves and order them by terminal load.
    matrix = np.asarray(tracks[:25], dtype=float)
    return matrix[np.argsort(matrix[:, -2])]


def save_trajectory_data(strategy: str, paths: list[tuple[np.ndarray, int]]) -> None:
    target = DATA_DIR / strategy / "trajectory"
    target.mkdir(parents=True, exist_ok=True)
    for index, (path, _) in enumerate(paths, start=1):
        with (target / f"bus_{index}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "station_position"])
            writer.writerows(path)


def save_load_data(strategy: str, matrix: np.ndarray) -> None:
    target = DATA_DIR / strategy
    target.mkdir(parents=True, exist_ok=True)
    with (target / "load_by_station.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bus_id", *[f"s{i}" for i in range(1, 40)]])
        for index, values in enumerate(matrix, start=1):
            writer.writerow([index, *np.round(values, 3)])


def plot_trajectory(
    strategy: str,
    source_stem: str,
    raw_buses: list[dict[str, np.ndarray]] | None,
) -> Path:
    if raw_buses is None:
        paths = digitize_trajectory(FIGURE_DIR / f"{source_stem}车辆轨迹.png")
    else:
        paths = [
            (np.column_stack([bus["time"], bus["loc"]]), index)
            for index, bus in enumerate(raw_buses)
        ]
    save_trajectory_data(strategy, paths)

    fig, ax = plt.subplots(figsize=(10.2, 7.1), constrained_layout=True)
    for path, color_index in paths:
        visible = (
            (path[:, 0] >= 0)
            & (path[:, 0] <= 12000)
            & (path[:, 1] >= 1)
            & (path[:, 1] <= 39)
        )
        ax.plot(
            path[visible, 0],
            path[visible, 1],
            color=PLOT_COLORS[color_index % len(PLOT_COLORS)],
            linewidth=1.45,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.set_xlim(0, 12000)
    ax.set_ylim(1, 39)
    ax.set_xticks(np.arange(0, 12001, 2000))
    ax.set_yticks(np.arange(1, 40, 3))
    ax.set_xlabel("时间/s", fontproperties=SONG)
    ax.set_ylabel("公交站点", fontproperties=SONG)
    style_axis(ax)
    output = OUT_DIR / f"fig_t34_trajectory_{strategy}.pdf"
    preview = PREVIEW_DIR / f"trajectory_{strategy}.png"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_load(
    strategy: str,
    source_stem: str,
    raw_buses: list[dict[str, np.ndarray]] | None,
) -> Path:
    if raw_buses is None:
        load_stem = "line_amappo_" if strategy == "改进MAPPO" else source_stem
        matrix = digitize_load(FIGURE_DIR / f"{load_stem}载客量.png")
    else:
        matrix = raw_load_by_station(raw_buses)
    save_load_data(strategy, matrix)

    fig, ax = plt.subplots(figsize=(10.2, 7.1), constrained_layout=True)
    stations = np.arange(1, 40)
    for index, values in enumerate(matrix):
        ax.plot(
            stations,
            values,
            color=PLOT_COLORS[index % len(PLOT_COLORS)],
            linewidth=1.65,
            alpha=0.88,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.set_xlim(0.5, 39.5)
    ax.set_xticks(np.arange(1, 40, 3))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("公交站点", fontproperties=SONG)
    ax.set_ylabel("载客量/人", fontproperties=SONG)
    style_axis(ax)
    output = OUT_DIR / f"fig_t34_load_{strategy}.pdf"
    preview = PREVIEW_DIR / f"load_{strategy}.png"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for strategy, source_stem, log_dir in STRATEGIES:
        # The generic log/trajectory folders are not the complete trial batch
        # used for the thesis figures. Use one consistent digitization route for
        # all six strategies so the comparison reproduces the reported results.
        raw_buses = None
        outputs.append(plot_trajectory(strategy, source_stem, raw_buses))
        outputs.append(plot_load(strategy, source_stem, raw_buses))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
