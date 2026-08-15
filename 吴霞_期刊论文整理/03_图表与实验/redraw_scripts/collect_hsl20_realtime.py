#!/usr/bin/env python3
"""Collect one or more complete Eira-to-Munkkivuori runs of HSL bus route 20."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2


FEED_URL = "https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl"
ROUTE_ID = "1020"
DIRECTION_ID = 0
TIMEZONE = ZoneInfo("Europe/Helsinki")
ORIGIN = {"stop_id": "1060101", "name": "Eira", "lat": 60.155613, "lon": 24.942476}
DESTINATION = {
    "stop_id": "1304161",
    "name": "Munkkivuoren ostosk.",
    "lat": 60.205420,
    "lon": 24.877250,
}
ORIGIN_RADIUS_M = 100.0
DESTINATION_RADIUS_M = 75.0
MIN_COMPLETE_DURATION_S = 15 * 60
MIN_COMPLETE_SAMPLES = 60

CSV_FIELDS = [
    "collection_received_at_utc",
    "feed_timestamp_utc",
    "vehicle_timestamp_utc",
    "vehicle_timestamp_helsinki",
    "route_id",
    "direction_id",
    "start_date",
    "start_time",
    "run_id",
    "vehicle_id",
    "latitude",
    "longitude",
    "bearing_deg",
    "speed_reported_mps",
    "speed_reported_kmh",
    "stop_id",
    "current_status_code",
    "current_status",
]

HEADWAY_FIELDS = [
    "sequence",
    "run_id",
    "vehicle_id",
    "start_date",
    "scheduled_trip_id",
    "scheduled_departure_eira",
    "scheduled_arrival_munkkivuori",
    "scheduled_headway_s",
    "scheduled_headway_min",
    "observed_origin_first_seen_helsinki",
    "observed_origin_departure_helsinki",
    "observed_destination_arrival_helsinki",
    "observed_headway_s",
    "observed_headway_min",
    "departure_delay_s",
    "travel_time_s",
    "sample_count",
]

STATUS_NAMES = {
    0: "INCOMING_AT",
    1: "STOPPED_AT",
    2: "IN_TRANSIT_TO",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Collect HSL route 20 direction 0 (Eira to Munkkivuori)."
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration-minutes", type=float, default=90.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--target-complete-runs",
        type=int,
        default=1,
        help="Stop after this many complete single-direction runs have been observed.",
    )
    parser.add_argument(
        "--timetable",
        type=Path,
        help="Current service-day timetable used to verify consecutive departures.",
    )
    parser.add_argument(
        "--require-consecutive",
        action="store_true",
        help="Stop only after the target is a gap-free block in the GTFS timetable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "03_图表与实验" / "source_data" / "hsl20_realtime",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.once and args.interval < 2:
        parser.error("--interval must be at least 2 seconds")
    if args.duration_minutes <= 0 or args.timeout <= 0 or args.target_complete_runs <= 0:
        parser.error("duration, timeout, and target-complete-runs must be positive")
    if args.require_consecutive and not args.timetable:
        parser.error("--timetable is required with --require-consecutive")
    return args


def utc_iso(timestamp: int | float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def helsinki_iso(timestamp: int | float) -> str:
    return datetime.fromtimestamp(timestamp, TIMEZONE).isoformat()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, value)))


def run_id(vehicle) -> str:
    trip = vehicle.trip
    vehicle_id = vehicle.vehicle.id or "unknown_vehicle"
    start_date = trip.start_date or "unknown_date"
    start_time = (trip.start_time or "unknown_time").replace(":", "")
    return f"{start_date}_{start_time}_{vehicle_id.replace('/', '-')}"


def decode_feed(payload: bytes, received_at: datetime) -> tuple[int, list[dict]]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(payload)
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        vehicle = entity.vehicle
        if vehicle.trip.route_id != ROUTE_ID or vehicle.trip.direction_id != DIRECTION_ID:
            continue
        timestamp = int(vehicle.timestamp or feed.header.timestamp or received_at.timestamp())
        speed_mps = float(vehicle.position.speed)
        status_code = int(vehicle.current_status)
        rows.append(
            {
                "collection_received_at_utc": received_at.isoformat(),
                "feed_timestamp_utc": utc_iso(feed.header.timestamp),
                "vehicle_timestamp_utc": utc_iso(timestamp),
                "vehicle_timestamp_helsinki": helsinki_iso(timestamp),
                "route_id": vehicle.trip.route_id,
                "direction_id": int(vehicle.trip.direction_id),
                "start_date": vehicle.trip.start_date,
                "start_time": vehicle.trip.start_time,
                "run_id": run_id(vehicle),
                "vehicle_id": vehicle.vehicle.id,
                "latitude": float(vehicle.position.latitude),
                "longitude": float(vehicle.position.longitude),
                "bearing_deg": float(vehicle.position.bearing),
                "speed_reported_mps": speed_mps,
                "speed_reported_kmh": speed_mps * 3.6,
                "stop_id": vehicle.stop_id,
                "current_status_code": status_code,
                "current_status": STATUS_NAMES.get(status_code, "UNKNOWN"),
            }
        )
    return int(feed.header.timestamp), rows


def append_json_line(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def count_existing_requests(path: Path) -> tuple[int, int]:
    total = 0
    successful = 0
    if not path.exists():
        return total, successful
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            try:
                successful += json.loads(line).get("status") == "ok"
            except json.JSONDecodeError:
                continue
    return total, successful


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in CSV_FIELDS} for row in rows)


def read_timetable(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(
        rows,
        key=lambda row: (
            row["service_date"],
            normalize_gtfs_time(row["departure_eira"]),
        ),
    )


def normalize_gtfs_time(value: str) -> str:
    hour, minute, second = (int(part) for part in value.split(":"))
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def service_datetime(service_date: str, gtfs_time: str) -> datetime:
    hour, minute, second = (int(part) for part in gtfs_time.split(":"))
    midnight = datetime.strptime(service_date, "%Y%m%d").replace(tzinfo=TIMEZONE)
    return midnight + timedelta(hours=hour, minutes=minute, seconds=second)


def select_consecutive_complete_runs(
    complete_runs: set[str],
    coverage: dict,
    timetable: list[dict[str, str]],
    target: int,
) -> list[str]:
    completed_by_departure = {
        (state["start_date"], normalize_gtfs_time(state["start_time"])): run
        for run, state in coverage.items()
        if run in complete_runs and state.get("start_date") and state.get("start_time")
    }
    if len(completed_by_departure) < target:
        return []
    for start in range(len(timetable) - target + 1):
        window = timetable[start : start + target]
        run_ids = [
            completed_by_departure.get(
                (row["service_date"], normalize_gtfs_time(row["departure_eira"]))
            )
            for row in window
        ]
        if all(run_ids):
            return [str(run_id) for run_id in run_ids]
    return []


def build_headway_rows(
    selected_runs: list[str],
    coverage: dict,
    timetable: list[dict[str, str]],
) -> list[dict]:
    schedule_by_departure = {
        (row["service_date"], normalize_gtfs_time(row["departure_eira"])): row
        for row in timetable
    }
    rows = []
    previous_scheduled = None
    previous_observed = None
    for sequence, run in enumerate(selected_runs, start=1):
        state = coverage[run]
        schedule = schedule_by_departure[
            (state["start_date"], normalize_gtfs_time(state["start_time"]))
        ]
        scheduled = service_datetime(schedule["service_date"], schedule["departure_eira"])
        observed = datetime.fromtimestamp(state["origin_departure_epoch"], TIMEZONE)
        origin_seen = datetime.fromtimestamp(state["origin_seen_epoch"], TIMEZONE)
        destination = datetime.fromtimestamp(state["destination_seen_epoch"], TIMEZONE)
        scheduled_headway = (
            (scheduled - previous_scheduled).total_seconds() if previous_scheduled else ""
        )
        observed_headway = (
            (observed - previous_observed).total_seconds() if previous_observed else ""
        )
        rows.append(
            {
                "sequence": sequence,
                "run_id": run,
                "vehicle_id": state["vehicle_id"],
                "start_date": state["start_date"],
                "scheduled_trip_id": schedule["trip_id"],
                "scheduled_departure_eira": schedule["departure_eira"],
                "scheduled_arrival_munkkivuori": schedule["arrival_munkkivuori"],
                "scheduled_headway_s": scheduled_headway,
                "scheduled_headway_min": scheduled_headway / 60 if scheduled_headway != "" else "",
                "observed_origin_first_seen_helsinki": origin_seen.isoformat(),
                "observed_origin_departure_helsinki": observed.isoformat(),
                "observed_destination_arrival_helsinki": destination.isoformat(),
                "observed_headway_s": observed_headway,
                "observed_headway_min": observed_headway / 60 if observed_headway != "" else "",
                "departure_delay_s": (observed - scheduled).total_seconds(),
                "travel_time_s": (destination - observed).total_seconds(),
                "sample_count": state["sample_count"],
            }
        )
        previous_scheduled = scheduled
        previous_observed = observed
    return rows


def write_headways(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADWAY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_existing_positions(path: Path) -> tuple[set[tuple[str, str]], dict, set[str], int]:
    seen_samples: set[tuple[str, str]] = set()
    coverage: dict = {}
    complete_runs: set[str] = set()
    count = 0
    if not path.exists() or path.stat().st_size == 0:
        return seen_samples, coverage, complete_runs, count
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["latitude"] = float(row["latitude"])
            row["longitude"] = float(row["longitude"])
            key = (row["run_id"], row["vehicle_timestamp_utc"])
            if key in seen_samples:
                continue
            seen_samples.add(key)
            count += 1
            if update_coverage(coverage, row):
                complete_runs.add(row["run_id"])
    return seen_samples, coverage, complete_runs, count


def update_coverage(coverage: dict, row: dict) -> bool:
    state = coverage.setdefault(
        row["run_id"],
        {
            "vehicle_id": row["vehicle_id"],
            "start_date": row["start_date"],
            "start_time": row["start_time"],
            "first_timestamp_utc": row["vehicle_timestamp_utc"],
            "last_timestamp_utc": row["vehicle_timestamp_utc"],
            "sample_count": 0,
            "stops_observed": set(),
            "origin_seen_epoch": None,
            "origin_last_seen_epoch": None,
            "origin_departure_epoch": None,
            "destination_seen_epoch": None,
            "destination_stop_id_seen": False,
            "minimum_origin_distance_m": float("inf"),
            "minimum_destination_distance_m": float("inf"),
        },
    )
    timestamp = datetime.fromisoformat(row["vehicle_timestamp_utc"]).timestamp()
    origin_distance = haversine_m(row["latitude"], row["longitude"], ORIGIN["lat"], ORIGIN["lon"])
    destination_distance = haversine_m(
        row["latitude"], row["longitude"], DESTINATION["lat"], DESTINATION["lon"]
    )
    state["sample_count"] += 1
    state["last_timestamp_utc"] = row["vehicle_timestamp_utc"]
    state["minimum_origin_distance_m"] = min(state["minimum_origin_distance_m"], origin_distance)
    state["minimum_destination_distance_m"] = min(
        state["minimum_destination_distance_m"], destination_distance
    )
    if row["stop_id"]:
        state["stops_observed"].add(row["stop_id"])

    at_origin = row["stop_id"] == ORIGIN["stop_id"] or origin_distance <= ORIGIN_RADIUS_M
    if at_origin and state["origin_seen_epoch"] is None:
        state["origin_seen_epoch"] = timestamp
    if at_origin:
        state["origin_last_seen_epoch"] = timestamp
    elif state["origin_last_seen_epoch"] is not None and state["origin_departure_epoch"] is None:
        state["origin_departure_epoch"] = timestamp

    elapsed = timestamp - state["origin_seen_epoch"] if state["origin_seen_epoch"] else 0
    if row["stop_id"] == DESTINATION["stop_id"]:
        state["destination_stop_id_seen"] = True

    # Route 20 passes close to its terminus before making the Munkkivuori loop.
    # HSL then announces the final stop while approaching it and may clear
    # stop_id at physical arrival.  Require the ordered combination of a final
    # stop announcement followed by entry into a small terminus radius.
    at_destination = (
        state["destination_stop_id_seen"]
        and destination_distance <= DESTINATION_RADIUS_M
    )
    if at_destination and state["origin_seen_epoch"] and elapsed >= MIN_COMPLETE_DURATION_S:
        state["destination_seen_epoch"] = timestamp

    return bool(
        state["origin_seen_epoch"]
        and state["origin_departure_epoch"]
        and state["destination_seen_epoch"]
        and state["sample_count"] >= MIN_COMPLETE_SAMPLES
    )


def serializable_coverage(coverage: dict) -> dict:
    result = {}
    for key, value in coverage.items():
        row = dict(value)
        row["stops_observed"] = sorted(row["stops_observed"])
        for name in ["minimum_origin_distance_m", "minimum_destination_distance_m"]:
            row[name] = round(row[name], 1) if math.isfinite(row[name]) else None
        result[key] = row
    return result


def write_summary(
    json_path: Path,
    markdown_path: Path,
    *,
    started_at: datetime,
    request_count: int,
    successful_requests: int,
    existing_sample_count: int,
    new_sample_count: int,
    complete_runs: set[str],
    selected_consecutive_runs: list[str],
    target_complete_runs: int,
    require_consecutive: bool,
    coverage: dict,
) -> None:
    first_observed = min(
        (value["first_timestamp_utc"] for value in coverage.values()), default=None
    )
    last_observed = max(
        (value["last_timestamp_utc"] for value in coverage.values()), default=None
    )
    ordered_complete_runs = sorted(
        complete_runs,
        key=lambda run: (
            coverage.get(run, {}).get("start_date", ""),
            coverage.get(run, {}).get("start_time", ""),
            run,
        ),
    )
    target_reached = (
        len(selected_consecutive_runs) >= target_complete_runs
        if require_consecutive
        else len(ordered_complete_runs) >= target_complete_runs
    )
    summary = {
        "source": FEED_URL,
        "route_id": ROUTE_ID,
        "direction_id": DIRECTION_ID,
        "direction": f"{ORIGIN['name']} -> {DESTINATION['name']}",
        "execution_started_at_utc": started_at.isoformat(),
        "execution_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_first_vehicle_timestamp_utc": first_observed,
        "observed_last_vehicle_timestamp_utc": last_observed,
        "request_count": request_count,
        "successful_requests": successful_requests,
        "existing_sample_count": existing_sample_count,
        "new_sample_count": new_sample_count,
        "total_position_samples": existing_sample_count + new_sample_count,
        "target_complete_runs": target_complete_runs,
        "complete_run_id": ordered_complete_runs[0] if ordered_complete_runs else None,
        "complete_run_ids": ordered_complete_runs,
        "complete_run_count": len(ordered_complete_runs),
        "require_consecutive": require_consecutive,
        "selected_consecutive_run_ids": selected_consecutive_runs,
        "selected_consecutive_run_count": len(selected_consecutive_runs),
        "target_reached": target_reached,
        "complete": bool(ordered_complete_runs),
        "runs": serializable_coverage(coverage),
        "speed_note": (
            "speed_reported_kmh is supplied by HSL GTFS-RT; position-derived interval speed "
            "is calculated separately during reconstruction."
        ),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "## HSL route 20 single-direction capture",
        "",
        f"- Direction: `{summary['direction']}` (`route_id={ROUTE_ID}`, `direction_id={DIRECTION_ID}`)",
        f"- Requests: {request_count} ({successful_requests} successful)",
        f"- Existing position samples resumed: {existing_sample_count}",
        f"- New position samples: {new_sample_count}",
        f"- Observed runs: {len(coverage)}",
        f"- Complete runs: {len(ordered_complete_runs)} / {target_complete_runs}",
        f"- Consecutive target required: {'yes' if require_consecutive else 'no'}",
        f"- Consecutive runs selected: {len(selected_consecutive_runs)} / {target_complete_runs}",
        (
            "- Selected run IDs: "
            + ", ".join(f"`{run}`" for run in selected_consecutive_runs)
            if selected_consecutive_runs
            else "- Selected run IDs: not yet available"
        ),
        (
            "- Complete run IDs: " + ", ".join(f"`{run}`" for run in ordered_complete_runs)
            if ordered_complete_runs
            else "- Complete run IDs: not yet observed"
        ),
        "- API key required: no",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def request_feed(timeout: float) -> bytes:
    request = urllib.request.Request(FEED_URL, headers={"User-Agent": "HSL20-research-collector/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def self_test() -> None:
    assert round(haversine_m(ORIGIN["lat"], ORIGIN["lon"], ORIGIN["lat"], ORIGIN["lon"]), 6) == 0
    coverage: dict = {}
    completed: set[str] = set()
    base = 1_786_700_000
    timetable = []
    for index in range(11):
        departure = f"08:{index * 10:02d}:00" if index < 6 else f"09:{(index - 6) * 10:02d}:00"
        timetable.append(
            {
                "service_date": "20260814",
                "trip_id": f"scheduled-{index + 1}",
                "departure_eira": departure,
                "arrival_munkkivuori": "09:00:00",
            }
        )

    def complete_test_run(index: int) -> None:
        run = f"test-{index + 1}"
        departure = timetable[index]["departure_eira"]
        departure_epoch = base + index * 600
        origin_row = {
            "run_id": run, "vehicle_id": f"22/{index + 1}", "start_date": "20260814",
            "start_time": departure, "vehicle_timestamp_utc": utc_iso(departure_epoch - 30),
            "latitude": ORIGIN["lat"], "longitude": ORIGIN["lon"], "stop_id": ORIGIN["stop_id"],
        }
        assert not update_coverage(coverage, origin_row)
        departed_row = dict(origin_row)
        departed_row.update(
            {
                "vehicle_timestamp_utc": utc_iso(departure_epoch),
                "latitude": 60.158,
                "longitude": 24.940,
                "stop_id": "",
            }
        )
        assert not update_coverage(coverage, departed_row)
        coverage[run]["sample_count"] = MIN_COMPLETE_SAMPLES
        destination_row = dict(origin_row)
        destination_row.update(
            {"vehicle_timestamp_utc": utc_iso(departure_epoch + MIN_COMPLETE_DURATION_S),
             "latitude": DESTINATION["lat"], "longitude": DESTINATION["lon"],
             "stop_id": DESTINATION["stop_id"]}
        )
        if update_coverage(coverage, destination_row):
            completed.add(run)

    for index in [*range(9), 10]:
        complete_test_run(index)
    assert not select_consecutive_complete_runs(completed, coverage, timetable, 10)
    complete_test_run(9)
    selected = select_consecutive_complete_runs(completed, coverage, timetable, 10)
    assert selected == [f"test-{index}" for index in range(1, 11)]
    headways = build_headway_rows(selected, coverage, timetable)
    assert len(headways) == 10
    assert all(row["observed_headway_s"] == 600 for row in headways[1:])
    print("HSL 20 realtime collector self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    service_day = started_at.astimezone(TIMEZONE).strftime("%Y%m%d")
    prefix = f"hsl20_eira_to_munkkivuori_{service_day}"
    positions_path = args.output_dir / f"{prefix}_raw_positions.csv"
    snapshots_path = args.output_dir / f"{prefix}_filtered_snapshots.jsonl"
    log_path = args.output_dir / f"{prefix}_collection_log.jsonl"
    protobuf_path = args.output_dir / f"{prefix}_first_feed.pb"
    summary_json = args.output_dir / f"{prefix}_collection_summary.json"
    summary_md = args.output_dir / f"{prefix}_collection_summary.md"
    headways_path = args.output_dir / f"{prefix}_headways.csv"
    timetable = read_timetable(args.timetable) if args.timetable else []

    seen_samples, coverage, complete_runs, existing_sample_count = load_existing_positions(
        positions_path
    )
    selected_consecutive_runs = (
        select_consecutive_complete_runs(
            complete_runs, coverage, timetable, args.target_complete_runs
        )
        if args.require_consecutive
        else []
    )
    existing_request_count, existing_successful_requests = count_existing_requests(log_path)
    request_count = 0
    successful_requests = 0
    new_sample_count = 0
    consecutive_errors = 0
    deadline = time.monotonic() + args.duration_minutes * 60

    try:
        while True:
            request_count += 1
            request_started = time.monotonic()
            try:
                payload = request_feed(args.timeout)
                received_at = datetime.now(timezone.utc)
                feed_timestamp, rows = decode_feed(payload, received_at)
                successful_requests += 1
                consecutive_errors = 0
                if not protobuf_path.exists():
                    protobuf_path.write_bytes(payload)

                fresh_rows = []
                for row in rows:
                    key = (row["run_id"], row["vehicle_timestamp_utc"])
                    if key in seen_samples:
                        continue
                    seen_samples.add(key)
                    fresh_rows.append(row)
                    if update_coverage(coverage, row):
                        complete_runs.add(row["run_id"])
                if args.require_consecutive:
                    selected_consecutive_runs = select_consecutive_complete_runs(
                        complete_runs, coverage, timetable, args.target_complete_runs
                    )
                append_csv(positions_path, fresh_rows)
                new_sample_count += len(fresh_rows)
                append_json_line(
                    snapshots_path,
                    {
                        "received_at_utc": received_at.isoformat(),
                        "feed_timestamp": feed_timestamp,
                        "route_id": ROUTE_ID,
                        "direction_id": DIRECTION_ID,
                        "vehicles": rows,
                    },
                )
                append_json_line(
                    log_path,
                    {
                        "received_at_utc": received_at.isoformat(),
                        "request": request_count,
                        "status": "ok",
                        "vehicles": len(rows),
                        "new_samples": len(fresh_rows),
                        "complete_run_ids": sorted(complete_runs),
                        "complete_run_count": len(complete_runs),
                        "selected_consecutive_run_ids": selected_consecutive_runs,
                        "consecutive_target_reached": (
                            len(selected_consecutive_runs) >= args.target_complete_runs
                        ),
                    },
                )
                print(
                    f"{received_at.isoformat()} request={request_count} "
                    f"vehicles={len(rows)} new_samples={len(fresh_rows)} "
                    f"complete_runs={len(complete_runs)} "
                    f"consecutive_runs={len(selected_consecutive_runs)}/{args.target_complete_runs}",
                    flush=True,
                )
            except Exception as error:  # retain an auditable log and tolerate brief outages
                consecutive_errors += 1
                append_json_line(
                    log_path,
                    {
                        "received_at_utc": datetime.now(timezone.utc).isoformat(),
                        "request": request_count,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                print(f"request={request_count} error={type(error).__name__}: {error}", flush=True)
                if args.once or consecutive_errors >= 10:
                    break

            target_reached = (
                len(selected_consecutive_runs) >= args.target_complete_runs
                if args.require_consecutive
                else len(complete_runs) >= args.target_complete_runs
            )
            if args.once or target_reached or time.monotonic() >= deadline:
                break
            elapsed = time.monotonic() - request_started
            time.sleep(max(0.0, args.interval - elapsed))
    finally:
        write_summary(
            summary_json,
            summary_md,
            started_at=started_at,
            request_count=existing_request_count + request_count,
            successful_requests=existing_successful_requests + successful_requests,
            existing_sample_count=existing_sample_count,
            new_sample_count=new_sample_count,
            complete_runs=complete_runs,
            selected_consecutive_runs=selected_consecutive_runs,
            target_complete_runs=args.target_complete_runs,
            require_consecutive=args.require_consecutive,
            coverage=coverage,
        )
        if selected_consecutive_runs:
            write_headways(
                headways_path,
                build_headway_rows(selected_consecutive_runs, coverage, timetable),
            )

    if successful_requests == 0:
        return 3
    if args.once:
        return 0
    target_reached = (
        len(selected_consecutive_runs) >= args.target_complete_runs
        if args.require_consecutive
        else len(complete_runs) >= args.target_complete_runs
    )
    return 0 if target_reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
