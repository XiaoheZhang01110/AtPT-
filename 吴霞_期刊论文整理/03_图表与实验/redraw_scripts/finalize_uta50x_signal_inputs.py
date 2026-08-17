#!/usr/bin/env python3
"""Match official ATSPM phases to the 50X movement using observed bus passages."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from uta50x_common import DATA_DIR, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select route-serving signal phases for UTA 50X.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prefix")
    parser.add_argument("--tolerance-seconds", type=float, default=10.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def detect_prefix(data_dir: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    files = sorted(data_dir.glob("uta50x_wvc_to_murray_*_metadata.json"))
    if not files:
        raise FileNotFoundError("No UTA 50X metadata file found")
    return files[-1].name.removesuffix("_metadata.json")


def epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def match(args: argparse.Namespace) -> dict:
    prefix = detect_prefix(args.data_dir, args.prefix)
    passages_path = args.data_dir / f"{prefix}_signal_passage_times.csv"
    windows_path = args.data_dir / f"{prefix}_official_green_windows_all_phases.csv"
    if not passages_path.exists() or not windows_path.exists():
        raise FileNotFoundError("Both signal passages and official ATSPM green windows are required")
    passages = read_csv(passages_path)
    windows = read_csv(windows_path)
    passages_by_signal: dict[str, list[float]] = defaultdict(list)
    for row in passages:
        passages_by_signal[row["udot_signal_id"]].append(epoch(row["passage_time_local"]))
    windows_by_candidate: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in windows:
        key = (row["udot_signal_id"], row["controller_phase"], row["chart_panel_index"])
        windows_by_candidate[key].append(row)

    candidate_rows: list[dict[str, object]] = []
    best_by_signal: dict[str, dict[str, object]] = {}
    for (signal_id, phase, panel), candidate_windows in windows_by_candidate.items():
        times = passages_by_signal.get(signal_id, [])
        intervals = [(epoch(row["green_start_local"]), epoch(row["green_end_local"])) for row in candidate_windows]
        matched = sum(
            any(start - args.tolerance_seconds <= passage <= end + args.tolerance_seconds for start, end in intervals)
            for passage in times
        )
        ratio = matched / len(times) if times else 0.0
        candidate = {
            "udot_signal_id": signal_id,
            "controller_phase": phase,
            "chart_panel_index": panel,
            "bus_passage_count": len(times),
            "matched_passage_count": matched,
            "match_ratio": f"{ratio:.4f}",
            "green_window_count": len(intervals),
        }
        candidate_rows.append(candidate)
        previous = best_by_signal.get(signal_id)
        score = (matched, ratio, len(intervals))
        previous_score = (
            int(previous["matched_passage_count"]),
            float(previous["match_ratio"]),
            int(previous["green_window_count"]),
        ) if previous else (-1, -1.0, -1)
        if score > previous_score:
            best_by_signal[signal_id] = candidate

    candidate_rows.sort(key=lambda row: (row["udot_signal_id"], int(row["controller_phase"]), int(row["chart_panel_index"])))
    write_csv(
        args.data_dir / f"{prefix}_signal_phase_matching_candidates.csv",
        candidate_rows,
        [
            "udot_signal_id",
            "controller_phase",
            "chart_panel_index",
            "bus_passage_count",
            "matched_passage_count",
            "match_ratio",
            "green_window_count",
        ],
    )

    selected_windows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    for signal_id, best in best_by_signal.items():
        chosen = windows_by_candidate[(signal_id, str(best["controller_phase"]), str(best["chart_panel_index"]))]
        chosen.sort(key=lambda row: epoch(row["green_start_local"]))
        starts = [epoch(row["green_start_local"]) for row in chosen]
        durations = [float(row["green_duration_s"]) for row in chosen]
        cycles = [right - left for left, right in zip(starts, starts[1:]) if 20 <= right - left <= 240]
        confidence = "high" if float(best["match_ratio"]) >= 0.8 and int(best["bus_passage_count"]) >= 5 else (
            "medium" if float(best["match_ratio"]) >= 0.5 else "low"
        )
        for row in chosen:
            selected_windows.append(
                {
                    **row,
                    "phase_match_ratio": best["match_ratio"],
                    "phase_match_confidence": confidence,
                }
            )
        plan_rows.append(
            {
                "udot_signal_id": signal_id,
                "route_distance_km": chosen[0]["route_distance_km"],
                "selected_controller_phase": best["controller_phase"],
                "selected_chart_panel_index": best["chart_panel_index"],
                "bus_passage_count": best["bus_passage_count"],
                "matched_passage_count": best["matched_passage_count"],
                "phase_match_ratio": best["match_ratio"],
                "phase_match_confidence": confidence,
                "observed_green_window_count": len(chosen),
                "median_cycle_s": f"{statistics.median(cycles):.2f}" if cycles else "",
                "mean_cycle_s": f"{statistics.fmean(cycles):.2f}" if cycles else "",
                "median_green_s": f"{statistics.median(durations):.2f}" if durations else "",
                "mean_green_s": f"{statistics.fmean(durations):.2f}" if durations else "",
                "timing_type": "observed actuated ATSPM intervals",
            }
        )
    selected_windows.sort(key=lambda row: (float(row["route_distance_km"]), epoch(row["green_start_local"])))
    plan_rows.sort(key=lambda row: float(row["route_distance_km"]))
    selected_fields = [
        "signal_index",
        "udot_signal_id",
        "route_distance_km",
        "controller_phase",
        "chart_panel_index",
        "green_window_index",
        "green_start_local",
        "green_end_local",
        "green_duration_s",
        "chart_file",
        "seconds_per_pixel",
        "source",
        "method",
        "phase_match_ratio",
        "phase_match_confidence",
    ]
    write_csv(args.data_dir / f"{prefix}_optimization_green_windows.csv", selected_windows, selected_fields)
    plan_fields = [
        "udot_signal_id",
        "route_distance_km",
        "selected_controller_phase",
        "selected_chart_panel_index",
        "bus_passage_count",
        "matched_passage_count",
        "phase_match_ratio",
        "phase_match_confidence",
        "observed_green_window_count",
        "median_cycle_s",
        "mean_cycle_s",
        "median_green_s",
        "mean_green_s",
        "timing_type",
    ]
    write_csv(args.data_dir / f"{prefix}_optimization_signal_plans.csv", plan_rows, plan_fields)
    summary = {
        "prefix": prefix,
        "signal_passage_count": len(passages),
        "candidate_phase_panel_count": len(candidate_rows),
        "selected_signal_count": len(plan_rows),
        "high_confidence_signal_count": sum(row["phase_match_confidence"] == "high" for row in plan_rows),
        "medium_confidence_signal_count": sum(row["phase_match_confidence"] == "medium" for row in plan_rows),
        "low_confidence_signal_count": sum(row["phase_match_confidence"] == "low" for row in plan_rows),
        "caveat": "Phase selection is inferred by matching bus passage times to official ATSPM windows; retain confidence labels in analysis.",
    }
    write_json(args.data_dir / f"{prefix}_signal_phase_match_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def self_test() -> None:
    assert epoch("2026-08-17T06:30:00-06:00") == epoch("2026-08-17T12:30:00+00:00")
    print("UTA 50X phase-matching self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    match(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
