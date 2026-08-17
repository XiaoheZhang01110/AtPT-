#!/usr/bin/env python3
"""Clean UTA 50X positions into model-ready trajectories, speeds, and headways."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from uta50x_common import DATA_DIR, gtfs_time_seconds, project_to_route, read_csv, write_csv, write_json


TIMEZONE = ZoneInfo("America/Denver")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct UTA 50X optimization inputs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prefix")
    parser.add_argument("--interval-seconds", type=int, default=1)
    parser.add_argument("--max-route-offset-m", type=float, default=120.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def detect_prefix(data_dir: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    files = sorted(data_dir.glob("uta50x_wvc_to_murray_*_metadata.json"))
    if not files:
        raise FileNotFoundError("No UTA 50X metadata file found")
    return files[-1].name.removesuffix("_metadata.json")


def parse_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def isotonic(values: list[float]) -> list[float]:
    """Unweighted pool-adjacent-violators regression."""
    blocks: list[list[float]] = []  # mean, weight, start, end
    for index, value in enumerate(values):
        blocks.append([value, 1.0, float(index), float(index)])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            blocks.append(
                [
                    (left[0] * left[1] + right[0] * right[1]) / weight,
                    weight,
                    left[2],
                    right[3],
                ]
            )
    output = [0.0] * len(values)
    for mean, _, start, end in blocks:
        for index in range(int(start), int(end) + 1):
            output[index] = mean
    return output


def linear_interpolate(xs: list[float], ys: list[float], target: float) -> float:
    if target <= xs[0]:
        return ys[0]
    if target >= xs[-1]:
        return ys[-1]
    index = bisect.bisect_right(xs, target) - 1
    fraction = (target - xs[index]) / (xs[index + 1] - xs[index])
    return ys[index] + fraction * (ys[index + 1] - ys[index])


def crossing_time(times: list[float], distances: list[float], target_distance: float) -> float | None:
    if not distances or target_distance < distances[0] or target_distance > distances[-1]:
        return None
    index = bisect.bisect_left(distances, target_distance)
    if index == 0:
        return times[0]
    d0, d1 = distances[index - 1], distances[index]
    if d1 == d0:
        return times[index]
    fraction = (target_distance - d0) / (d1 - d0)
    return times[index - 1] + fraction * (times[index] - times[index - 1])


def schedule_epoch(service_date: str, gtfs_time: str) -> float:
    day = date(int(service_date[:4]), int(service_date[4:6]), int(service_date[6:8]))
    local_midnight = datetime.combine(day, datetime.min.time(), tzinfo=TIMEZONE)
    return (local_midnight + timedelta(seconds=gtfs_time_seconds(gtfs_time))).timestamp()


def build_trajectory(
    rows: list[dict[str, str]],
    interval: int,
    max_offset_m: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    accepted: dict[int, tuple[float, float]] = {}
    for row in rows:
        if float(row["route_offset_m"]) > max_offset_m:
            continue
        timestamp = int(round(parse_timestamp(row["vehicle_timestamp_utc"])))
        distance = float(row["route_distance_km"])
        offset = float(row["route_offset_m"])
        previous = accepted.get(timestamp)
        if previous is None or offset < previous[1]:
            accepted[timestamp] = (distance, offset)
    if len(accepted) < 10:
        return [], {"accepted_raw_samples": len(accepted), "complete": False}
    raw_times = sorted(accepted)
    raw_distances = [accepted[timestamp][0] for timestamp in raw_times]
    fitted = isotonic(raw_distances)
    grid = list(range(raw_times[0], raw_times[-1] + 1, interval))
    distances = [linear_interpolate(raw_times, fitted, timestamp) for timestamp in grid]
    half_window = max(2, int(round(5 / interval)))
    speeds: list[float] = []
    for index in range(len(grid)):
        left = max(0, index - half_window)
        right = min(len(grid) - 1, index + half_window)
        delta_t = grid[right] - grid[left]
        speed = 0.0 if delta_t <= 0 else (distances[right] - distances[left]) * 1000.0 / delta_t
        speeds.append(max(0.0, min(35.0, speed)))
    smooth_radius = max(1, int(round(5 / interval)))
    smooth_speeds = [
        statistics.fmean(speeds[max(0, i - smooth_radius) : min(len(speeds), i + smooth_radius + 1)])
        for i in range(len(speeds))
    ]
    output = [
        {
            "timestamp_epoch": timestamp,
            "timestamp_utc": datetime.fromtimestamp(timestamp, ZoneInfo("UTC")).isoformat(),
            "elapsed_s": timestamp - grid[0],
            "route_distance_km": distance,
            "speed_mps": speed,
            "speed_kmh": speed * 3.6,
        }
        for timestamp, distance, speed in zip(grid, distances, smooth_speeds)
    ]
    return output, {
        "accepted_raw_samples": len(accepted),
        "start_epoch": grid[0],
        "end_epoch": grid[-1],
        "duration_s": grid[-1] - grid[0],
        "min_distance_km": min(distances),
        "max_distance_km": max(distances),
    }


def interpolate_elevation(elevation: list[dict[str, str]], distance: float) -> float:
    xs = [float(row["route_distance_km"]) for row in elevation]
    ys = [float(row["elevation_m"]) for row in elevation]
    return linear_interpolate(xs, ys, distance)


def reconstruct(args: argparse.Namespace) -> dict:
    prefix = detect_prefix(args.data_dir, args.prefix)
    metadata = json.loads((args.data_dir / f"{prefix}_metadata.json").read_text(encoding="utf-8"))
    timetable = read_csv(args.data_dir / f"{prefix}_target10_timetable.csv")
    target_index = {row["trip_id"]: int(row["target_run_index"]) for row in timetable}
    positions = read_csv(args.data_dir / f"{prefix}_raw_vehicle_positions.csv")
    grouped: dict[str, list[dict[str, str]]] = {trip_id: [] for trip_id in target_index}
    for row in positions:
        if row["trip_id"] in grouped:
            grouped[row["trip_id"]].append(row)

    trajectory_by_trip: dict[str, list[dict[str, object]]] = {}
    quality: dict[str, dict[str, object]] = {}
    trajectory_rows: list[dict[str, object]] = []
    for trip_id, index in sorted(target_index.items(), key=lambda item: item[1]):
        trajectory, summary = build_trajectory(
            grouped[trip_id], args.interval_seconds, args.max_route_offset_m
        )
        if trajectory:
            summary["complete"] = bool(
                summary["min_distance_km"] <= metadata["route_length_km"] * 0.08
                and summary["max_distance_km"] >= metadata["route_length_km"] * 0.92
                and summary["duration_s"] >= 600
            )
            trajectory_by_trip[trip_id] = trajectory
            for row in trajectory:
                trajectory_rows.append(
                    {"run_index": index, "trip_id": trip_id, **row}
                )
        quality[trip_id] = summary
    if not trajectory_rows:
        raise RuntimeError("No usable UTA 50X trajectories were reconstructed")
    write_csv(
        args.data_dir / f"{prefix}_clean_spacetime_trajectories.csv",
        trajectory_rows,
        [
            "run_index",
            "trip_id",
            "timestamp_epoch",
            "timestamp_utc",
            "elapsed_s",
            "route_distance_km",
            "speed_mps",
            "speed_kmh",
        ],
    )

    observed_rows: list[dict[str, object]] = []
    for scheduled in sorted(timetable, key=lambda row: int(row["target_run_index"])):
        trip_id = scheduled["trip_id"]
        trajectory = trajectory_by_trip.get(trip_id, [])
        if trajectory:
            times = [float(row["timestamp_epoch"]) for row in trajectory]
            distances = [float(row["route_distance_km"]) for row in trajectory]
            departure = crossing_time(times, distances, min(0.20, metadata["route_length_km"] * 0.02))
            arrival = crossing_time(times, distances, metadata["route_length_km"] - min(0.20, metadata["route_length_km"] * 0.02))
        else:
            departure = arrival = None
        scheduled_departure = schedule_epoch(metadata["service_date"], scheduled["scheduled_origin_departure"])
        scheduled_arrival = schedule_epoch(metadata["service_date"], scheduled["scheduled_destination_arrival"])
        observed_rows.append(
            {
                "run_index": scheduled["target_run_index"],
                "trip_id": trip_id,
                "scheduled_origin_departure_local": scheduled["scheduled_origin_departure"],
                "observed_origin_departure_utc": datetime.fromtimestamp(departure, ZoneInfo("UTC")).isoformat() if departure else "",
                "origin_departure_delay_s": round(departure - scheduled_departure, 1) if departure else "",
                "scheduled_destination_arrival_local": scheduled["scheduled_destination_arrival"],
                "observed_destination_arrival_utc": datetime.fromtimestamp(arrival, ZoneInfo("UTC")).isoformat() if arrival else "",
                "destination_arrival_delay_s": round(arrival - scheduled_arrival, 1) if arrival else "",
                "observed_runtime_s": round(arrival - departure, 1) if departure and arrival else "",
                "observation_method": "trajectory crossing at 0.20 km from each terminus",
                "complete": quality.get(trip_id, {}).get("complete", False),
            }
        )
    write_csv(args.data_dir / f"{prefix}_observed_timetable.csv", observed_rows, list(observed_rows[0]))

    headway_rows: list[dict[str, object]] = []
    valid_departures = [row for row in observed_rows if row["observed_origin_departure_utc"]]
    for previous, current in zip(valid_departures, valid_departures[1:]):
        previous_epoch = parse_timestamp(str(previous["observed_origin_departure_utc"]))
        current_epoch = parse_timestamp(str(current["observed_origin_departure_utc"]))
        headway_rows.append(
            {
                "leading_run_index": previous["run_index"],
                "following_run_index": current["run_index"],
                "leading_trip_id": previous["trip_id"],
                "following_trip_id": current["trip_id"],
                "observed_headway_s": round(current_epoch - previous_epoch, 1),
                "observed_headway_min": round((current_epoch - previous_epoch) / 60.0, 3),
            }
        )
    write_csv(
        args.data_dir / f"{prefix}_headways.csv",
        headway_rows,
        [
            "leading_run_index",
            "following_run_index",
            "leading_trip_id",
            "following_trip_id",
            "observed_headway_s",
            "observed_headway_min",
        ],
    )

    distance_step = 0.05
    distance_grid = [
        index * distance_step
        for index in range(int(math.floor(metadata["route_length_km"] / distance_step)) + 1)
    ]
    if distance_grid[-1] < metadata["route_length_km"]:
        distance_grid.append(metadata["route_length_km"])
    mean_speed_rows: list[dict[str, object]] = []
    for distance in distance_grid:
        speed_values = []
        for trajectory in trajectory_by_trip.values():
            xs = [float(row["route_distance_km"]) for row in trajectory]
            if xs[0] <= distance <= xs[-1]:
                speed_values.append(
                    linear_interpolate(xs, [float(row["speed_mps"]) for row in trajectory], distance)
                )
        mean_speed_rows.append(
            {
                "route_distance_km": f"{distance:.6f}",
                "mean_speed_mps": f"{statistics.fmean(speed_values):.6f}" if speed_values else "",
                "mean_speed_kmh": f"{statistics.fmean(speed_values) * 3.6:.6f}" if speed_values else "",
                "contributing_runs": len(speed_values),
            }
        )
    write_csv(args.data_dir / f"{prefix}_mean_speed_curve.csv", mean_speed_rows, list(mean_speed_rows[0]))

    shape = read_csv(args.data_dir / f"{prefix}_shape.csv")
    route_points = [(float(row["latitude"]), float(row["longitude"])) for row in shape]
    route_distances = [float(row["route_distance_km"]) for row in shape]
    elevation = read_csv(args.data_dir / f"{prefix}_elevation_426.csv")
    stops = read_csv(args.data_dir / f"{prefix}_stops.csv")
    signals_path = args.data_dir / f"{prefix}_udot_signals.csv"
    signals = read_csv(signals_path) if signals_path.exists() else []
    node_rows: list[dict[str, object]] = []
    stop_distances: list[tuple[dict[str, str], float]] = []
    for stop in stops:
        distance, offset = project_to_route(
            float(stop["latitude"]), float(stop["longitude"]), route_points, route_distances
        )
        stop_distances.append((stop, distance))
        node_rows.append(
            {
                "node_type": "bus_stop",
                "node_id": stop["stop_id"],
                "node_name": stop["stop_name"],
                "route_distance_km": f"{distance:.6f}",
                "elevation_m": f"{interpolate_elevation(elevation, distance):.3f}",
                "latitude": stop["latitude"],
                "longitude": stop["longitude"],
                "route_offset_m": f"{offset:.2f}",
            }
        )
    for signal in signals:
        distance = float(signal["route_distance_km"])
        name = " / ".join(
            value for value in (signal["street_east_west"], signal["street_north_south"]) if value
        )
        node_rows.append(
            {
                "node_type": "traffic_signal",
                "node_id": signal["udot_signal_id"],
                "node_name": name,
                "route_distance_km": f"{distance:.6f}",
                "elevation_m": f"{interpolate_elevation(elevation, distance):.3f}",
                "latitude": signal["latitude"],
                "longitude": signal["longitude"],
                "route_offset_m": signal["route_offset_m"],
            }
        )
    node_rows.sort(key=lambda row: (float(row["route_distance_km"]), row["node_type"]))
    write_csv(args.data_dir / f"{prefix}_optimization_nodes.csv", node_rows, list(node_rows[0]))

    dwell_rows: list[dict[str, object]] = []
    for trip_id, trajectory in trajectory_by_trip.items():
        for stop, stop_distance in stop_distances:
            candidates = [
                int(row["timestamp_epoch"])
                for row in trajectory
                if abs(float(row["route_distance_km"]) - stop_distance) <= 0.060
                and float(row["speed_mps"]) <= 1.0
            ]
            groups: list[list[int]] = []
            for timestamp in candidates:
                if not groups or timestamp - groups[-1][-1] > max(5, args.interval_seconds * 3):
                    groups.append([timestamp])
                else:
                    groups[-1].append(timestamp)
            valid_groups = [group for group in groups if group[-1] - group[0] + args.interval_seconds >= 3]
            if not valid_groups:
                continue
            dwell = max(valid_groups, key=lambda group: group[-1] - group[0])
            dwell_rows.append(
                {
                    "run_index": target_index[trip_id],
                    "trip_id": trip_id,
                    "stop_sequence": stop["stop_sequence"],
                    "stop_id": stop["stop_id"],
                    "stop_name": stop["stop_name"],
                    "route_distance_km": f"{stop_distance:.6f}",
                    "observed_arrival_utc": datetime.fromtimestamp(dwell[0], ZoneInfo("UTC")).isoformat(),
                    "observed_departure_utc": datetime.fromtimestamp(dwell[-1] + args.interval_seconds, ZoneInfo("UTC")).isoformat(),
                    "dwell_time_s": dwell[-1] - dwell[0] + args.interval_seconds,
                    "detection_rule": "speed <= 1.0 m/s within 60 m of GTFS stop",
                }
            )
    write_csv(
        args.data_dir / f"{prefix}_observed_stop_dwells.csv",
        dwell_rows,
        [
            "run_index",
            "trip_id",
            "stop_sequence",
            "stop_id",
            "stop_name",
            "route_distance_km",
            "observed_arrival_utc",
            "observed_departure_utc",
            "dwell_time_s",
            "detection_rule",
        ],
    )

    signal_passages: list[dict[str, object]] = []
    for signal in signals:
        distance = float(signal["route_distance_km"])
        for trip_id, trajectory in trajectory_by_trip.items():
            times = [float(row["timestamp_epoch"]) for row in trajectory]
            distances = [float(row["route_distance_km"]) for row in trajectory]
            passage = crossing_time(times, distances, distance)
            if passage is not None:
                signal_passages.append(
                    {
                        "signal_index": signal["signal_index"],
                        "udot_signal_id": signal["udot_signal_id"],
                        "route_distance_km": signal["route_distance_km"],
                        "run_index": target_index[trip_id],
                        "trip_id": trip_id,
                        "passage_time_utc": datetime.fromtimestamp(passage, ZoneInfo("UTC")).isoformat(),
                        "passage_time_local": datetime.fromtimestamp(passage, TIMEZONE).isoformat(),
                    }
                )
    write_csv(
        args.data_dir / f"{prefix}_signal_passage_times.csv",
        signal_passages,
        [
            "signal_index",
            "udot_signal_id",
            "route_distance_km",
            "run_index",
            "trip_id",
            "passage_time_utc",
            "passage_time_local",
        ],
    )

    summary = {
        "prefix": prefix,
        "target_trip_count": len(timetable),
        "reconstructed_trip_count": len(trajectory_by_trip),
        "complete_trip_count": sum(bool(item.get("complete")) for item in quality.values()),
        "headway_count": len(headway_rows),
        "expected_headway_count": max(0, len(timetable) - 1),
        "observed_stop_dwell_count": len(dwell_rows),
        "signal_passage_count": len(signal_passages),
        "quality_by_trip": quality,
    }
    write_json(args.data_dir / f"{prefix}_reconstruction_summary.json", summary)
    if summary["complete_trip_count"] != len(timetable) or summary["headway_count"] != len(timetable) - 1:
        raise RuntimeError(
            f"Dataset incomplete: {summary['complete_trip_count']}/{len(timetable)} trips and "
            f"{summary['headway_count']}/{len(timetable)-1} headways"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def self_test() -> None:
    assert isotonic([0.0, 2.0, 1.0, 3.0]) == [0.0, 1.5, 1.5, 3.0]
    assert linear_interpolate([0.0, 2.0], [0.0, 4.0], 1.0) == 2.0
    assert crossing_time([0.0, 10.0], [0.0, 1.0], 0.5) == 5.0
    print("UTA 50X reconstruction self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    reconstruct(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
