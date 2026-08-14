#!/usr/bin/env python3
"""Reconstruct and plot the complete A148 speed profile from API snapshots.

Figure contract
---------------
Conclusion: the official A148 position stream reveals the detailed spatial
variation of operating speed over one complete outbound-and-return run.
Evidence: a single hero curve plots reconstructed speed against cumulative
travel distance, with raw interval estimates retained as faint points and the
Express Bus Terminal turnaround identified explicitly.
Archetype: quantitative single-panel hero figure.
Export: publication PDF with editable text plus reviewable CSV source data.
Review risks: irregular API refresh intervals, missing snapshots, coordinate
jumps, and the distinction between interval-average and controller speed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-wuxia-a148-speed")
)

# Codex installs plotting dependencies into this workspace-local directory.
# A normal Python environment with matplotlib installed simply ignores it.
LOCAL_PACKAGES = Path(__file__).resolve().parents[3] / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_008.8
TURNAROUND_SECTION = 21
EXPECTED_ONE_WAY_KM = 22.1
MAX_PLAUSIBLE_SPEED_KMH = 75.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Reconstruct the A148 speed curve from collected API snapshots."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="a148_raw_positions_YYYYMMDD.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root
        / "03_图表与实验"
        / "source_data"
        / "a148_reconstructed_speed_profile.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=root
        / "03_图表与实验"
        / "redrawn_figures"
        / "Figure_A148_reconstructed_speed_profile.pdf",
    )
    parser.add_argument("--vehicle-id", default="")
    parser.add_argument("--max-gap-seconds", type=float, default=60.0)
    parser.add_argument("--max-speed-kmh", type=float, default=MAX_PLAUSIBLE_SPEED_KMH)
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require observations near both termini before exporting.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the PDF figure; use --no-plot on a font-limited cloud runner.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    font_dir = Path("/System/Library/Fonts/Supplemental")
    for font_file in font_dir.glob("Times New Roman*.ttf"):
        mpl.font_manager.fontManager.addfont(font_file)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.linewidth": 1.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 20,
            "legend.frameon": False,
        }
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def select_lon_lat(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    candidates = [("tmX", "tmY"), ("posX", "posY")]
    for x_name, y_name in candidates:
        if x_name not in frame or y_name not in frame:
            continue
        x = numeric(frame[x_name])
        y = numeric(frame[y_name])
        plausible = x.between(124, 132) & y.between(33, 39)
        if plausible.mean() >= 0.8:
            return x.where(plausible), y.where(plausible)
    raise ValueError("No plausible WGS84 longitude/latitude pair found in tmX/tmY or posX/posY.")


def haversine_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def parse_api_time(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    parsed = pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return parsed.fillna(fallback)


def choose_vehicle(frame: pd.DataFrame, requested: str) -> tuple[pd.DataFrame, str]:
    frame["vehId"] = frame["vehId"].astype("string")
    if requested:
        chosen = requested
        subset = frame.loc[frame["vehId"] == requested].copy()
        if subset.empty:
            raise ValueError(f"Vehicle {requested} is not present in the input file.")
        return subset, chosen
    counts = frame.groupby("vehId")["dataTm"].nunique().sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No vehicle observations found.")
    chosen = str(counts.index[0])
    return frame.loc[frame["vehId"] == chosen].copy(), chosen


def hampel_replace(values: pd.Series, radius: int = 2, threshold: float = 3.0) -> pd.Series:
    output = values.copy()
    array = values.to_numpy(dtype=float)
    for index, value in enumerate(array):
        if not np.isfinite(value):
            continue
        left = max(0, index - radius)
        right = min(len(array), index + radius + 1)
        window = array[left:right]
        window = window[np.isfinite(window)]
        if len(window) < 3:
            continue
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0 and abs(value - median) > threshold * robust_sigma:
            output.iloc[index] = median
    return output


def reconstruct(
    raw: pd.DataFrame,
    vehicle_id: str = "",
    max_gap_seconds: float = 60.0,
    max_speed_kmh: float = MAX_PLAUSIBLE_SPEED_KMH,
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    required = {"dataTm", "vehId", "sectOrd", "stopFlag"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame, chosen_vehicle = choose_vehicle(raw.copy(), vehicle_id)
    frame["timestamp_seoul"] = parse_api_time(frame["dataTm"])
    frame["sectOrd"] = numeric(frame["sectOrd"])
    frame["stopFlag"] = numeric(frame["stopFlag"]).fillna(0)
    frame["longitude"], frame["latitude"] = select_lon_lat(frame)
    frame = (
        frame.dropna(subset=["timestamp_seoul", "longitude", "latitude", "sectOrd"])
        .sort_values("timestamp_seoul")
        .drop_duplicates(subset=["timestamp_seoul"], keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 3:
        raise ValueError("At least three distinct API timestamps are required.")

    frame["elapsed_s"] = (frame["timestamp_seoul"] - frame["timestamp_seoul"].iloc[0]).dt.total_seconds()
    frame["dt_s"] = frame["timestamp_seoul"].diff().dt.total_seconds()
    step_distance = np.full(len(frame), np.nan, dtype=float)
    step_distance[1:] = haversine_m(
        frame["longitude"].to_numpy()[:-1],
        frame["latitude"].to_numpy()[:-1],
        frame["longitude"].to_numpy()[1:],
        frame["latitude"].to_numpy()[1:],
    )
    frame["step_distance_m"] = step_distance
    frame["section_step"] = frame["sectOrd"].diff()
    frame["gap_flag"] = frame["dt_s"].gt(max_gap_seconds)
    frame["reverse_flag"] = frame["section_step"].lt(0)
    frame["speed_raw_kmh"] = frame["step_distance_m"] / frame["dt_s"] * 3.6
    frame["speed_outlier_flag"] = (
        frame["speed_raw_kmh"].gt(max_speed_kmh)
        | frame["speed_raw_kmh"].lt(0)
        | frame["gap_flag"]
        | frame["reverse_flag"]
    )
    frame.loc[frame["speed_outlier_flag"], "speed_raw_kmh"] = np.nan
    frame.loc[frame["stopFlag"].eq(1), "speed_raw_kmh"] = 0.0
    invalid_interval = frame["gap_flag"] | frame["reverse_flag"]
    frame.loc[invalid_interval, "speed_raw_kmh"] = np.nan

    frame["speed_reconstructed_kmh"] = hampel_replace(frame["speed_raw_kmh"])
    frame["speed_reconstructed_kmh"] = frame["speed_reconstructed_kmh"].interpolate(
        method="linear", limit=2, limit_area="inside"
    )
    frame.loc[frame["stopFlag"].eq(1), "speed_reconstructed_kmh"] = 0.0
    # A row represents the interval ending at that timestamp.  Never invent an
    # interval-average speed across a long API outage or a reversed section
    # jump, even when the missing value is isolated between valid samples.
    frame.loc[invalid_interval, "speed_reconstructed_kmh"] = np.nan

    # Keep cumulative distance auditable: invalid jumps and long gaps are not integrated.
    valid_step = frame["step_distance_m"].where(~frame["speed_outlier_flag"], 0.0).fillna(0.0)
    frame["cumulative_distance_raw_km"] = valid_step.cumsum() / 1000.0
    frame["direction"] = np.where(
        frame["sectOrd"].le(TURNAROUND_SECTION), "Outbound", "Inbound"
    )

    # Scale only when both direction endpoints are observed.  This corrects the
    # chord-length underestimation caused by sparse API updates while preserving
    # the local speed estimates computed from unscaled displacement.
    frame["cumulative_distance_km"] = frame["cumulative_distance_raw_km"]
    outbound = frame["direction"].eq("Outbound")
    inbound = frame["direction"].eq("Inbound")
    outbound_complete = frame.loc[outbound, "sectOrd"].min() <= 2 and frame.loc[outbound, "sectOrd"].max() >= 20
    inbound_complete = inbound.any() and frame.loc[inbound, "sectOrd"].min() <= 23 and frame.loc[inbound, "sectOrd"].max() >= 40
    if outbound_complete:
        start = frame.loc[outbound, "cumulative_distance_raw_km"].min()
        end = frame.loc[outbound, "cumulative_distance_raw_km"].max()
        if end > start:
            frame.loc[outbound, "cumulative_distance_km"] = (
                frame.loc[outbound, "cumulative_distance_raw_km"] - start
            ) * EXPECTED_ONE_WAY_KM / (end - start)
    if inbound_complete:
        start = frame.loc[inbound, "cumulative_distance_raw_km"].min()
        end = frame.loc[inbound, "cumulative_distance_raw_km"].max()
        if end > start:
            frame.loc[inbound, "cumulative_distance_km"] = EXPECTED_ONE_WAY_KM + (
                frame.loc[inbound, "cumulative_distance_raw_km"] - start
            ) * EXPECTED_ONE_WAY_KM / (end - start)

    valid_speeds = frame["speed_reconstructed_kmh"].dropna()
    diagnostics = {
        "sample_count": float(len(frame)),
        "valid_speed_count": float(len(valid_speeds)),
        "median_interval_s": float(frame["dt_s"].dropna().median()),
        "max_interval_s": float(frame["dt_s"].dropna().max()),
        "outlier_count": float(frame["speed_outlier_flag"].sum()),
        "observed_distance_km": float(frame["cumulative_distance_raw_km"].max()),
        "mean_speed_kmh": float(valid_speeds.mean()),
        "max_speed_kmh": float(valid_speeds.max()),
        "outbound_complete": float(bool(outbound_complete)),
        "inbound_complete": float(bool(inbound_complete)),
    }
    return frame, chosen_vehicle, diagnostics


def require_complete(frame: pd.DataFrame, diagnostics: dict[str, float]) -> None:
    if not diagnostics["outbound_complete"] or not diagnostics["inbound_complete"]:
        section_min = int(frame["sectOrd"].min())
        section_max = int(frame["sectOrd"].max())
        raise ValueError(
            "The collected trajectory is incomplete: observed section range "
            f"{section_min}–{section_max}. A complete export requires both directions."
        )


def plot_profile(frame: pd.DataFrame, output_pdf: Path, vehicle_id: str) -> None:
    configure_style()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.0, 7.2), constrained_layout=True)

    colors = {"Outbound": "#1769AA", "Inbound": "#D97706"}
    for direction in ("Outbound", "Inbound"):
        subset = frame.loc[frame["direction"].eq(direction)]
        if subset.empty:
            continue
        ax.scatter(
            subset["cumulative_distance_km"],
            subset["speed_raw_kmh"],
            s=14,
            color=colors[direction],
            alpha=0.22,
            linewidths=0,
            rasterized=False,
        )
        ax.plot(
            subset["cumulative_distance_km"],
            subset["speed_reconstructed_kmh"],
            color=colors[direction],
            linewidth=2.5,
            label=direction,
        )

    stop_rows = frame.loc[frame["stopFlag"].eq(1)].drop_duplicates(subset=["sectOrd"])
    ax.scatter(
        stop_rows["cumulative_distance_km"],
        np.zeros(len(stop_rows)),
        marker="|",
        s=180,
        linewidths=1.4,
        color="#222222",
        zorder=5,
        label="Observed stop event",
    )
    ax.axvline(EXPECTED_ONE_WAY_KM, color="#555555", linestyle="--", linewidth=1.5)
    ax.text(
        EXPECTED_ONE_WAY_KM,
        0.82,
        "Express Bus Terminal\nturnaround",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=20,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xlim(0, 2 * EXPECTED_ONE_WAY_KM)
    finite_speed = frame["speed_reconstructed_kmh"].dropna()
    y_max = max(50.0, math.ceil((finite_speed.max() + 5.0) / 10.0) * 10.0)
    ax.set_ylim(-1.5, y_max)
    ax.set_xlabel("Cumulative distance along the round trip (km)")
    ax.set_ylabel("Reconstructed operating speed (km h$^{-1}$)")
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right", ncol=3, columnspacing=1.4, handlelength=2.2)
    ax.text(
        0.01,
        0.98,
        f"Vehicle ID: {vehicle_id}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        color="#333333",
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def run_self_test() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-08-12 03:30:00",
            "2026-08-12 03:30:12",
            "2026-08-12 03:30:24",
            "2026-08-12 03:30:36",
            "2026-08-12 03:32:36",
            "2026-08-12 03:32:48",
            "2026-08-12 03:33:00",
            "2026-08-12 03:33:12",
        ]
    )
    raw = pd.DataFrame(
        {
            "dataTm": timestamps.strftime("%Y%m%d%H%M%S"),
            "vehId": ["test"] * 8,
            "sectOrd": [1, 2, 3, 20, 22, 23, 40, 41],
            "stopFlag": [1, 0, 0, 0, 0, 0, 0, 1],
            "tmX": np.linspace(127.0730, 127.0800, 8),
            "tmY": np.linspace(37.6600, 37.6530, 8),
            "posX": [""] * 8,
            "posY": [""] * 8,
        }
    )
    profile, vehicle, diagnostics = reconstruct(raw, max_speed_kmh=300.0)
    assert vehicle == "test"
    assert len(profile) == 8
    assert profile["cumulative_distance_raw_km"].is_monotonic_increasing
    assert diagnostics["outbound_complete"] == 1.0
    assert diagnostics["inbound_complete"] == 1.0
    assert profile.loc[4, "gap_flag"]
    assert pd.isna(profile.loc[4, "speed_raw_kmh"])
    assert pd.isna(profile.loc[4, "speed_reconstructed_kmh"])
    test_pdf = Path(tempfile.gettempdir()) / "a148_speed_self_test.pdf"
    plot_profile(profile, test_pdf, vehicle)
    assert test_pdf.exists() and test_pdf.stat().st_size > 1000
    print("Self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.input is None:
        raise SystemExit("An input CSV is required unless --self-test is used.")
    if not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")

    raw = pd.read_csv(args.input, dtype="string")
    profile, vehicle_id, diagnostics = reconstruct(
        raw,
        vehicle_id=args.vehicle_id,
        max_gap_seconds=args.max_gap_seconds,
        max_speed_kmh=args.max_speed_kmh,
    )
    if args.require_complete:
        require_complete(profile, diagnostics)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    if args.plot:
        plot_profile(profile, args.output_pdf, vehicle_id)

    print(f"Vehicle: {vehicle_id}")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")
    print(f"Source data: {args.output_csv}")
    if args.plot:
        print(f"Figure: {args.output_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
