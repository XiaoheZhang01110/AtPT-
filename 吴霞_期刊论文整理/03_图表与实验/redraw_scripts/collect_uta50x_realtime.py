#!/usr/bin/env python3
"""Capture ten consecutive UTA 50X trips from official GTFS-Realtime feeds."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from google.transit import gtfs_realtime_pb2

from uta50x_common import DATA_DIR, project_to_route, read_csv, request_bytes, write_json


VEHICLE_URL = "https://apps.rideuta.com/tms/gtfs/Vehicle"
TRIP_UPDATE_URL = "https://apps.rideuta.com/tms/gtfs/TripUpdate"

POSITION_FIELDS = [
    "request_index",
    "collected_at_utc",
    "feed_timestamp_utc",
    "trip_id",
    "route_id",
    "direction_id",
    "vehicle_id",
    "vehicle_label",
    "latitude",
    "longitude",
    "bearing_deg",
    "reported_speed_mps",
    "current_stop_sequence",
    "stop_id",
    "current_status",
    "vehicle_timestamp_utc",
    "route_distance_km",
    "route_offset_m",
]

UPDATE_FIELDS = [
    "request_index",
    "collected_at_utc",
    "feed_timestamp_utc",
    "trip_id",
    "route_id",
    "direction_id",
    "vehicle_id",
    "stop_sequence",
    "stop_id",
    "arrival_time_utc",
    "arrival_delay_s",
    "departure_time_utc",
    "departure_delay_s",
    "schedule_relationship",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture official UTA 50X GTFS-RT data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prefix", help="Static file prefix; auto-detected when omitted")
    parser.add_argument("--duration-minutes", type=float, default=240.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--target-trips", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def detect_prefix(data_dir: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    files = sorted(data_dir.glob("uta50x_wvc_to_murray_*_metadata.json"))
    if not files:
        raise FileNotFoundError("No UTA 50X metadata file found")
    return files[-1].name.removesuffix("_metadata.json")


def epoch_iso(value: int) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value else ""


def load_feed(url: str, timeout: int) -> gtfs_realtime_pb2.FeedMessage:
    payload = request_bytes(url, timeout=timeout, attempts=2)
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(payload)
    return feed


def enum_name(message, field_name: str) -> str:
    descriptor = message.DESCRIPTOR.fields_by_name[field_name].enum_type
    value = getattr(message, field_name)
    return descriptor.values_by_number[value].name if value in descriptor.values_by_number else str(value)


def append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def completion_summary(
    samples: dict[str, list[tuple[float, int]]],
    route_length_km: float,
    target_ids: list[str],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for trip_id in target_ids:
        values = samples.get(trip_id, [])
        distances = [value[0] for value in values]
        timestamps = [value[1] for value in values]
        span_s = max(timestamps) - min(timestamps) if timestamps else 0
        complete = bool(
            len(values) >= 30
            and min(distances, default=route_length_km) <= route_length_km * 0.08
            and max(distances, default=0.0) >= route_length_km * 0.92
            and span_s >= 600
        )
        summary[trip_id] = {
            "sample_count": len(values),
            "min_route_distance_km": min(distances) if distances else None,
            "max_route_distance_km": max(distances) if distances else None,
            "observation_span_s": span_s,
            "complete": complete,
        }
    return summary


def run_capture(args: argparse.Namespace) -> int:
    prefix = detect_prefix(args.data_dir, args.prefix)
    metadata = json.loads((args.data_dir / f"{prefix}_metadata.json").read_text(encoding="utf-8"))
    timetable = read_csv(args.data_dir / f"{prefix}_target10_timetable.csv")[: args.target_trips]
    target_ids = [row["trip_id"] for row in timetable]
    target_set = set(target_ids)
    route_id = str(metadata["route_id"])
    direction_id = str(metadata["direction_id"])
    shape = read_csv(args.data_dir / f"{prefix}_shape.csv")
    route_points = [(float(row["latitude"]), float(row["longitude"])) for row in shape]
    route_distances = [float(row["route_distance_km"]) for row in shape]
    route_length = route_distances[-1]

    positions_path = args.data_dir / f"{prefix}_raw_vehicle_positions.csv"
    updates_path = args.data_dir / f"{prefix}_raw_trip_updates.csv"
    log_path = args.data_dir / f"{prefix}_collection_log.jsonl"
    samples: dict[str, list[tuple[float, int]]] = {trip_id: [] for trip_id in target_ids}
    seen_updates: set[tuple] = set()
    started = time.monotonic()
    request_index = 0
    successful_vehicle_requests = 0
    successful_update_requests = 0

    while True:
        request_index += 1
        collected_at = datetime.now(timezone.utc)
        status = "ok"
        error = ""
        new_positions = 0
        new_updates = 0
        try:
            vehicle_feed = load_feed(VEHICLE_URL, args.timeout)
            successful_vehicle_requests += 1
            position_rows: list[dict] = []
            for entity in vehicle_feed.entity:
                if not entity.HasField("vehicle"):
                    continue
                vehicle = entity.vehicle
                trip = vehicle.trip
                if trip.trip_id not in target_set:
                    continue
                if trip.route_id and trip.route_id != route_id:
                    continue
                if trip.HasField("direction_id") and str(trip.direction_id) != direction_id:
                    continue
                latitude = float(vehicle.position.latitude)
                longitude = float(vehicle.position.longitude)
                distance_km, offset_m = project_to_route(
                    latitude, longitude, route_points, route_distances
                )
                timestamp = int(vehicle.timestamp or vehicle_feed.header.timestamp or collected_at.timestamp())
                samples[trip.trip_id].append((distance_km, timestamp))
                position_rows.append(
                    {
                        "request_index": request_index,
                        "collected_at_utc": collected_at.isoformat(),
                        "feed_timestamp_utc": epoch_iso(int(vehicle_feed.header.timestamp)),
                        "trip_id": trip.trip_id,
                        "route_id": trip.route_id,
                        "direction_id": trip.direction_id if trip.HasField("direction_id") else "",
                        "vehicle_id": vehicle.vehicle.id,
                        "vehicle_label": vehicle.vehicle.label,
                        "latitude": f"{latitude:.7f}",
                        "longitude": f"{longitude:.7f}",
                        "bearing_deg": vehicle.position.bearing if vehicle.position.HasField("bearing") else "",
                        "reported_speed_mps": vehicle.position.speed if vehicle.position.HasField("speed") else "",
                        "current_stop_sequence": vehicle.current_stop_sequence,
                        "stop_id": vehicle.stop_id,
                        "current_status": enum_name(vehicle, "current_status"),
                        "vehicle_timestamp_utc": epoch_iso(timestamp),
                        "route_distance_km": f"{distance_km:.6f}",
                        "route_offset_m": f"{offset_m:.2f}",
                    }
                )
            append_rows(positions_path, POSITION_FIELDS, position_rows)
            new_positions = len(position_rows)

            update_feed = load_feed(TRIP_UPDATE_URL, args.timeout)
            successful_update_requests += 1
            update_rows: list[dict] = []
            for entity in update_feed.entity:
                if not entity.HasField("trip_update"):
                    continue
                update = entity.trip_update
                trip = update.trip
                if trip.trip_id not in target_set:
                    continue
                for stop_update in update.stop_time_update:
                    arrival_time = int(stop_update.arrival.time) if stop_update.HasField("arrival") else 0
                    departure_time = int(stop_update.departure.time) if stop_update.HasField("departure") else 0
                    arrival_delay = stop_update.arrival.delay if stop_update.HasField("arrival") and stop_update.arrival.HasField("delay") else ""
                    departure_delay = stop_update.departure.delay if stop_update.HasField("departure") and stop_update.departure.HasField("delay") else ""
                    key = (
                        trip.trip_id,
                        stop_update.stop_sequence,
                        stop_update.stop_id,
                        arrival_time,
                        arrival_delay,
                        departure_time,
                        departure_delay,
                    )
                    if key in seen_updates:
                        continue
                    seen_updates.add(key)
                    update_rows.append(
                        {
                            "request_index": request_index,
                            "collected_at_utc": collected_at.isoformat(),
                            "feed_timestamp_utc": epoch_iso(int(update_feed.header.timestamp)),
                            "trip_id": trip.trip_id,
                            "route_id": trip.route_id,
                            "direction_id": trip.direction_id if trip.HasField("direction_id") else "",
                            "vehicle_id": update.vehicle.id,
                            "stop_sequence": stop_update.stop_sequence,
                            "stop_id": stop_update.stop_id,
                            "arrival_time_utc": epoch_iso(arrival_time),
                            "arrival_delay_s": arrival_delay,
                            "departure_time_utc": epoch_iso(departure_time),
                            "departure_delay_s": departure_delay,
                            "schedule_relationship": enum_name(stop_update, "schedule_relationship"),
                        }
                    )
            append_rows(updates_path, UPDATE_FIELDS, update_rows)
            new_updates = len(update_rows)
        except Exception as exc:
            status = "api_error"
            error = f"{type(exc).__name__}: {exc}"

        summary = completion_summary(samples, route_length, target_ids)
        complete_count = sum(bool(item["complete"]) for item in summary.values())
        log_entry = {
            "collected_at_utc": collected_at.isoformat(),
            "request_index": request_index,
            "status": status,
            "error": error,
            "new_position_rows": new_positions,
            "new_trip_update_rows": new_updates,
            "complete_trip_count": complete_count,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(json.dumps(log_entry, ensure_ascii=False), flush=True)

        if args.once:
            break
        if complete_count >= len(target_ids):
            break
        elapsed = time.monotonic() - started
        if elapsed >= args.duration_minutes * 60:
            break
        time.sleep(max(0.0, args.interval_seconds - (time.monotonic() - started - elapsed)))

    summary = completion_summary(samples, route_length, target_ids)
    result = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "prefix": prefix,
        "vehicle_feed": VEHICLE_URL,
        "trip_update_feed": TRIP_UPDATE_URL,
        "request_count": request_index,
        "successful_vehicle_requests": successful_vehicle_requests,
        "successful_trip_update_requests": successful_update_requests,
        "requested_target_trip_count": len(target_ids),
        "complete_trip_count": sum(bool(item["complete"]) for item in summary.values()),
        "trips": summary,
    }
    write_json(args.data_dir / f"{prefix}_capture_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.once:
        return 0 if successful_vehicle_requests and successful_update_requests else 2
    return 0 if result["complete_trip_count"] == len(target_ids) else 2


def self_test() -> None:
    assert epoch_iso(0) == ""
    sample = completion_summary({"t": [(0.0, 0), (10.0, 700)] * 15}, 10.0, ["t"])
    assert sample["t"]["complete"] is True
    print("UTA 50X real-time collector self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
