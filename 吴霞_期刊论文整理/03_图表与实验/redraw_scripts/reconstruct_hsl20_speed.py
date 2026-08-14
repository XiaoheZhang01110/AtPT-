#!/usr/bin/env python3
"""Map-match and reconstruct one HSL 20 Eira-to-Munkkivuori speed profile."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


EARTH_RADIUS_M = 6_371_008.8
MAX_SPEED_KMH = 75.0
MAX_GAP_S = 30.0
MAX_MAP_ERROR_M = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct one complete HSL route 20 speed curve.")
    parser.add_argument("raw_positions", type=Path, nargs="?")
    parser.add_argument("shape", type=Path, nargs="?")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def project(lat: float, lon: float, latitude_origin: float) -> tuple[float, float]:
    return (
        EARTH_RADIUS_M * math.radians(lon) * math.cos(math.radians(latitude_origin)),
        EARTH_RADIUS_M * math.radians(lat),
    )


def prepare_shape(rows: list[dict]) -> tuple[list[tuple[float, float]], list[float]]:
    rows = sorted(rows, key=lambda row: int(row["shape_pt_sequence"]))
    latitude_origin = sum(float(row["shape_pt_lat"]) for row in rows) / len(rows)
    points = [
        project(float(row["shape_pt_lat"]), float(row["shape_pt_lon"]), latitude_origin)
        for row in rows
    ]
    distances = [float(row["shape_dist_traveled_km"]) for row in rows]
    return points, distances


def nearest_route_position(
    point: tuple[float, float], shape: list[tuple[float, float]], distances: list[float]
) -> tuple[float, float]:
    px, py = point
    best_error_sq = float("inf")
    best_distance = 0.0
    for index in range(len(shape) - 1):
        ax, ay = shape[index]
        bx, by = shape[index + 1]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        fraction = 0.0 if length_sq == 0 else ((px - ax) * dx + (py - ay) * dy) / length_sq
        fraction = min(1.0, max(0.0, fraction))
        qx, qy = ax + fraction * dx, ay + fraction * dy
        error_sq = (px - qx) ** 2 + (py - qy) ** 2
        if error_sq < best_error_sq:
            best_error_sq = error_sq
            best_distance = distances[index] + fraction * (distances[index + 1] - distances[index])
    return best_distance, math.sqrt(best_error_sq)


def choose_run(groups: dict[str, list[dict]], requested: str, route_length: float) -> str:
    if requested:
        if requested not in groups:
            raise ValueError(f"Run {requested} is not present in the raw positions file")
        return requested
    scores = {}
    for run_id, rows in groups.items():
        route_values = [float(row["route_distance_km"]) for row in rows]
        span = max(route_values) - min(route_values)
        endpoint_bonus = route_length if min(route_values) <= 0.5 and max(route_values) >= route_length - 0.5 else 0
        scores[run_id] = endpoint_bonus + span
    if not scores:
        raise ValueError("No route-20 direction-0 observations found")
    return max(scores, key=scores.get)


def reconstruct(raw_rows: list[dict], shape_rows: list[dict], requested_run: str = ""):
    if len(shape_rows) < 2:
        raise ValueError("The route shape must contain at least two points")
    latitude_origin = sum(float(row["shape_pt_lat"]) for row in shape_rows) / len(shape_rows)
    shape, shape_distances = prepare_shape(shape_rows)
    route_length = shape_distances[-1]

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        if row.get("route_id") != "1020" or row.get("direction_id") != "0":
            continue
        position, error = nearest_route_position(
            project(float(row["latitude"]), float(row["longitude"]), latitude_origin),
            shape,
            shape_distances,
        )
        enriched = dict(row)
        enriched["route_distance_km"] = position
        enriched["map_match_error_m"] = error
        groups[row["run_id"]].append(enriched)

    selected = choose_run(groups, requested_run, route_length)
    rows = sorted(groups[selected], key=lambda row: row["vehicle_timestamp_utc"])
    deduplicated = []
    seen_timestamps = set()
    for row in rows:
        if row["vehicle_timestamp_utc"] in seen_timestamps:
            continue
        seen_timestamps.add(row["vehicle_timestamp_utc"])
        deduplicated.append(row)
    rows = deduplicated
    if len(rows) < 3:
        raise ValueError("At least three distinct vehicle timestamps are required")

    first_time = datetime.fromisoformat(rows[0]["vehicle_timestamp_utc"])
    previous = None
    valid_position_speeds = []
    for row in rows:
        timestamp = datetime.fromisoformat(row["vehicle_timestamp_utc"])
        row["elapsed_s"] = (timestamp - first_time).total_seconds()
        row["dt_s"] = ""
        row["route_step_km"] = ""
        row["speed_position_kmh"] = ""
        row["quality_flag"] = "ok"
        reported = float(row["speed_reported_kmh"])
        if not 0 <= reported <= MAX_SPEED_KMH:
            row["quality_flag"] = "reported_speed_outlier"
        if float(row["map_match_error_m"]) > MAX_MAP_ERROR_M:
            row["quality_flag"] = "map_match_error"

        if previous is not None:
            previous_time = datetime.fromisoformat(previous["vehicle_timestamp_utc"])
            dt = (timestamp - previous_time).total_seconds()
            step = float(row["route_distance_km"]) - float(previous["route_distance_km"])
            row["dt_s"] = dt
            row["route_step_km"] = step
            if dt <= 0:
                row["quality_flag"] = "nonpositive_time_step"
            elif dt > MAX_GAP_S:
                row["quality_flag"] = "long_gap"
            elif step < -0.05:
                row["quality_flag"] = "reverse_position_jump"
            else:
                position_speed = max(0.0, step / dt * 3600.0)
                if position_speed > MAX_SPEED_KMH:
                    row["quality_flag"] = "position_speed_outlier"
                elif row["quality_flag"] == "ok":
                    row["speed_position_kmh"] = position_speed
                    valid_position_speeds.append(position_speed)
        row["speed_selected_kmh"] = (
            reported if 0 <= reported <= MAX_SPEED_KMH else row["speed_position_kmh"]
        )
        previous = row

    minimum_position = min(float(row["route_distance_km"]) for row in rows)
    maximum_position = max(float(row["route_distance_km"]) for row in rows)
    elapsed = float(rows[-1]["elapsed_s"])
    complete = minimum_position <= 0.5 and maximum_position >= route_length - 0.5 and elapsed >= 15 * 60
    summary = {
        "run_id": selected,
        "vehicle_id": rows[0]["vehicle_id"],
        "start_date": rows[0]["start_date"],
        "start_time": rows[0]["start_time"],
        "sample_count": len(rows),
        "route_length_km": route_length,
        "minimum_route_distance_km": minimum_position,
        "maximum_route_distance_km": maximum_position,
        "elapsed_minutes": elapsed / 60,
        "complete": complete,
        "mean_reported_speed_kmh": sum(float(row["speed_reported_kmh"]) for row in rows) / len(rows),
        "mean_valid_position_speed_kmh": (
            sum(valid_position_speeds) / len(valid_position_speeds) if valid_position_speeds else None
        ),
        "maximum_map_match_error_m": max(float(row["map_match_error_m"]) for row in rows),
        "quality_flag_counts": dict(
            (flag, sum(row["quality_flag"] == flag for row in rows))
            for flag in sorted({row["quality_flag"] for row in rows})
        ),
    }
    return rows, summary


def write_outputs(rows: list[dict], summary: dict, output_csv: Path, output_summary: Path) -> None:
    fieldnames = list(rows[0])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    shape = [
        {"shape_pt_sequence": "1", "shape_pt_lat": "60.0", "shape_pt_lon": "24.0", "shape_dist_traveled_km": "0"},
        {"shape_pt_sequence": "2", "shape_pt_lat": "60.0", "shape_pt_lon": "24.01", "shape_dist_traveled_km": "0.556"},
    ]
    points, distances = prepare_shape(shape)
    position, error = nearest_route_position(points[1], points, distances)
    assert abs(position - 0.556) < 1e-6 and error < 1e-6
    print("HSL 20 speed reconstruction self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.raw_positions or not args.shape:
        raise SystemExit("raw_positions and shape are required unless --self-test is used")
    rows, summary = reconstruct(read_csv(args.raw_positions), read_csv(args.shape), args.run_id)
    output_csv = args.output_csv or args.raw_positions.with_name(
        args.raw_positions.name.replace("_raw_positions.csv", "_speed_profile.csv")
    )
    output_summary = args.output_summary or output_csv.with_name(
        output_csv.name.replace(".csv", "_summary.json")
    )
    if args.require_complete and not summary["complete"]:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit("Selected run is incomplete; retaining raw data without exporting a misleading curve")
    write_outputs(rows, summary, output_csv, output_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
