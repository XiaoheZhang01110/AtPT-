#!/usr/bin/env python3
"""Collect one complete Eira-to-Munkkivuori run of HSL bus route 20."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
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
        "--output-dir",
        type=Path,
        default=root / "03_图表与实验" / "source_data" / "hsl20_realtime",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.once and args.interval < 2:
        parser.error("--interval must be at least 2 seconds")
    if args.duration_minutes <= 0 or args.timeout <= 0:
        parser.error("duration and timeout must be positive")
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


def load_existing_positions(path: Path) -> tuple[set[tuple[str, str]], dict, str, int]:
    seen_samples: set[tuple[str, str]] = set()
    coverage: dict = {}
    complete_run = ""
    count = 0
    if not path.exists() or path.stat().st_size == 0:
        return seen_samples, coverage, complete_run, count
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["latitude"] = float(row["latitude"])
            row["longitude"] = float(row["longitude"])
            key = (row["run_id"], row["vehicle_timestamp_utc"])
            if key in seen_samples:
                continue
            seen_samples.add(key)
            count += 1
            if update_coverage(coverage, row) and not complete_run:
                complete_run = row["run_id"]
    return seen_samples, coverage, complete_run, count


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
    complete_run: str,
    coverage: dict,
) -> None:
    first_observed = min(
        (value["first_timestamp_utc"] for value in coverage.values()), default=None
    )
    last_observed = max(
        (value["last_timestamp_utc"] for value in coverage.values()), default=None
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
        "complete_run_id": complete_run or None,
        "complete": bool(complete_run),
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
        f"- Complete run: `{complete_run}`" if complete_run else "- Complete run: not yet observed",
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
    base = 1_786_700_000
    origin_row = {
        "run_id": "test", "vehicle_id": "22/test", "start_date": "20260814",
        "start_time": "08:00:00", "vehicle_timestamp_utc": utc_iso(base),
        "latitude": ORIGIN["lat"], "longitude": ORIGIN["lon"], "stop_id": ORIGIN["stop_id"],
    }
    assert not update_coverage(coverage, origin_row)
    coverage["test"]["sample_count"] = MIN_COMPLETE_SAMPLES
    destination_row = dict(origin_row)
    destination_row.update(
        {"vehicle_timestamp_utc": utc_iso(base + MIN_COMPLETE_DURATION_S),
         "latitude": DESTINATION["lat"], "longitude": DESTINATION["lon"],
         "stop_id": DESTINATION["stop_id"]}
    )
    assert update_coverage(coverage, destination_row)
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

    seen_samples, coverage, complete_run, existing_sample_count = load_existing_positions(
        positions_path
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
                    if update_coverage(coverage, row) and not complete_run:
                        complete_run = row["run_id"]
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
                        "complete_run_id": complete_run or None,
                    },
                )
                print(
                    f"{received_at.isoformat()} request={request_count} "
                    f"vehicles={len(rows)} new_samples={len(fresh_rows)} "
                    f"complete_run={complete_run or '-'}",
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

            if args.once or complete_run or time.monotonic() >= deadline:
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
            complete_run=complete_run,
            coverage=coverage,
        )

    if successful_requests == 0:
        return 3
    if args.once:
        return 0
    return 0 if complete_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
