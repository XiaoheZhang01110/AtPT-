#!/usr/bin/env python3
"""Download and extract one weekday direction of UTA route 50X (MVX)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from uta50x_common import (
    DATA_DIR,
    cumulative_distances_km,
    gtfs_time_seconds,
    request_bytes,
    write_csv,
    write_json,
)


GTFS_URLS = (
    "https://gtfsfeed.rideuta.com/GTFS.zip",
    "https://apps.rideuta.com/tms/gtfs/Static",
)
TIMEZONE = ZoneInfo("America/Denver")
ROUTE_SHORT_NAME = "50X"
DEFAULT_HEADSIGN_TOKEN = "MURRAY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UTA 50X static inputs and ten-trip timetable.")
    parser.add_argument("--gtfs-zip", type=Path)
    parser.add_argument("--service-date", help="YYYYMMDD; default is the next/current weekday in Utah")
    parser.add_argument("--headsign-token", default=DEFAULT_HEADSIGN_TOKEN)
    parser.add_argument("--first-departure", default="06:30:00")
    parser.add_argument("--trip-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def rows_from_zip(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        yield from csv.DictReader(text)


def default_service_date() -> str:
    day = datetime.now(TIMEZONE).date()
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.strftime("%Y%m%d")


def active_service_ids(archive: zipfile.ZipFile, service_date: str) -> set[str]:
    day = datetime.strptime(service_date, "%Y%m%d")
    weekday = day.strftime("%A").lower()
    active: set[str] = set()
    names = set(archive.namelist())
    if "calendar.txt" in names:
        active.update(
            row["service_id"]
            for row in rows_from_zip(archive, "calendar.txt")
            if row["start_date"] <= service_date <= row["end_date"] and row.get(weekday) == "1"
        )
    if "calendar_dates.txt" in names:
        for row in rows_from_zip(archive, "calendar_dates.txt"):
            if row["date"] != service_date:
                continue
            if row["exception_type"] == "1":
                active.add(row["service_id"])
            elif row["exception_type"] == "2":
                active.discard(row["service_id"])
    return active


def download_gtfs() -> tuple[Path, str]:
    errors: list[str] = []
    for index, url in enumerate(GTFS_URLS):
        try:
            payload = request_bytes(url, timeout=180, attempts=2)
            if not payload.startswith(b"PK"):
                raise RuntimeError("response is not a ZIP archive")
            target = Path(tempfile.gettempdir()) / f"uta_gtfs_50x_{index}.zip"
            target.write_bytes(payload)
            return target, url
        except Exception as exc:  # try the second official endpoint
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Both official UTA GTFS endpoints failed:\n" + "\n".join(errors))


def select_route_and_direction(
    archive: zipfile.ZipFile,
    active: set[str],
    headsign_token: str,
) -> tuple[dict[str, str], str, list[dict[str, str]]]:
    routes = [row for row in rows_from_zip(archive, "routes.txt") if row["route_short_name"].strip().upper() == ROUTE_SHORT_NAME]
    if len(routes) != 1:
        raise ValueError(f"Expected exactly one route with route_short_name=50X, found {len(routes)}")
    route = routes[0]
    candidates = [
        row
        for row in rows_from_zip(archive, "trips.txt")
        if row["route_id"] == route["route_id"] and row["service_id"] in active
    ]
    if not candidates:
        raise ValueError("No active route 50X trips for the requested service date")
    token = headsign_token.strip().upper()
    matching = [row for row in candidates if token in row.get("trip_headsign", "").upper()]
    if matching:
        direction_id = Counter(row.get("direction_id", "") for row in matching).most_common(1)[0][0]
    else:
        direction_id = Counter(row.get("direction_id", "") for row in candidates).most_common(1)[0][0]
    trips = [row for row in candidates if row.get("direction_id", "") == direction_id]
    return route, direction_id, trips


def extract(
    gtfs_zip: Path,
    service_date: str,
    output_dir: Path,
    headsign_token: str,
    first_departure: str,
    trip_count: int,
    source_url: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gtfs_zip) as archive:
        active = active_service_ids(archive, service_date)
        route, direction_id, trips = select_route_and_direction(archive, active, headsign_token)
        trip_by_id = {row["trip_id"]: row for row in trips}
        stop_times_by_trip: dict[str, list[dict[str, str]]] = {trip_id: [] for trip_id in trip_by_id}
        for row in rows_from_zip(archive, "stop_times.txt"):
            if row["trip_id"] in stop_times_by_trip:
                stop_times_by_trip[row["trip_id"]].append(row)
        valid_trips = []
        for trip in trips:
            values = stop_times_by_trip[trip["trip_id"]]
            values.sort(key=lambda row: int(row["stop_sequence"]))
            if values:
                valid_trips.append(trip)
        valid_trips.sort(key=lambda row: gtfs_time_seconds(stop_times_by_trip[row["trip_id"]][0]["departure_time"]))

        start_seconds = gtfs_time_seconds(first_departure)
        eligible = [
            row
            for row in valid_trips
            if gtfs_time_seconds(stop_times_by_trip[row["trip_id"]][0]["departure_time"]) >= start_seconds
        ]
        target_trips = eligible[:trip_count]
        if len(target_trips) != trip_count:
            raise ValueError(f"Only {len(target_trips)} direction-{direction_id} trips depart after {first_departure}; need {trip_count}")

        shape_id = Counter(row.get("shape_id", "") for row in target_trips).most_common(1)[0][0]
        reference_trip = next((row for row in target_trips if row.get("shape_id", "") == shape_id), target_trips[0])
        reference_times = stop_times_by_trip[reference_trip["trip_id"]]
        stop_ids = {row["stop_id"] for row in reference_times}
        stops = {row["stop_id"]: row for row in rows_from_zip(archive, "stops.txt") if row["stop_id"] in stop_ids}
        shape_source = [row for row in rows_from_zip(archive, "shapes.txt") if row["shape_id"] == shape_id]
        shape_source.sort(key=lambda row: int(row["shape_pt_sequence"]))

    route_points = [(float(row["shape_pt_lat"]), float(row["shape_pt_lon"])) for row in shape_source]
    if len(route_points) < 2:
        raise ValueError("The selected 50X shape has fewer than two points")
    route_distances = cumulative_distances_km(route_points)
    shape_rows = [
        {
            "shape_pt_sequence": row["shape_pt_sequence"],
            "latitude": f"{lat:.7f}",
            "longitude": f"{lon:.7f}",
            "route_distance_km": f"{distance:.6f}",
        }
        for row, (lat, lon), distance in zip(shape_source, route_points, route_distances)
    ]

    stop_rows = []
    for item in reference_times:
        stop = stops[item["stop_id"]]
        stop_rows.append(
            {
                "stop_sequence": item["stop_sequence"],
                "stop_id": item["stop_id"],
                "stop_code": stop.get("stop_code", ""),
                "stop_name": stop["stop_name"],
                "latitude": stop["stop_lat"],
                "longitude": stop["stop_lon"],
                "reference_arrival_time": item["arrival_time"],
                "reference_departure_time": item["departure_time"],
            }
        )

    schedule_rows = []
    for sequence, trip in enumerate(valid_trips, start=1):
        values = stop_times_by_trip[trip["trip_id"]]
        schedule_rows.append(
            {
                "service_date": service_date,
                "schedule_sequence": sequence,
                "trip_id": trip["trip_id"],
                "route_id": route["route_id"],
                "service_id": trip["service_id"],
                "direction_id": direction_id,
                "trip_headsign": trip.get("trip_headsign", ""),
                "shape_id": trip.get("shape_id", ""),
                "scheduled_origin_departure": values[0]["departure_time"],
                "scheduled_destination_arrival": values[-1]["arrival_time"],
            }
        )
    target_ids = {trip["trip_id"] for trip in target_trips}
    target_rows = [dict(row) for row in schedule_rows if row["trip_id"] in target_ids]
    target_rows.sort(key=lambda row: gtfs_time_seconds(row["scheduled_origin_departure"]))
    for index, row in enumerate(target_rows, start=1):
        row["target_run_index"] = index

    target_stop_times = []
    for index, trip in enumerate(target_trips, start=1):
        for value in stop_times_by_trip[trip["trip_id"]]:
            stop = stops.get(value["stop_id"], {})
            target_stop_times.append(
                {
                    "target_run_index": index,
                    "trip_id": trip["trip_id"],
                    "stop_sequence": value["stop_sequence"],
                    "stop_id": value["stop_id"],
                    "stop_name": stop.get("stop_name", ""),
                    "scheduled_arrival_time": value["arrival_time"],
                    "scheduled_departure_time": value["departure_time"],
                }
            )

    prefix = f"uta50x_wvc_to_murray_{service_date}"
    write_csv(output_dir / f"{prefix}_shape.csv", shape_rows, list(shape_rows[0]))
    write_csv(output_dir / f"{prefix}_stops.csv", stop_rows, list(stop_rows[0]))
    write_csv(output_dir / f"{prefix}_weekday_timetable.csv", schedule_rows, list(schedule_rows[0]))
    write_csv(output_dir / f"{prefix}_target10_timetable.csv", target_rows, list(target_rows[0]))
    write_csv(output_dir / f"{prefix}_target10_stop_times.csv", target_stop_times, list(target_stop_times[0]))

    metadata = {
        "prepared_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source_url,
        "service_date": service_date,
        "timezone": "America/Denver",
        "route_id": route["route_id"],
        "route_short_name": route["route_short_name"],
        "route_long_name": route.get("route_long_name", ""),
        "direction_id": direction_id,
        "origin": {"stop_id": stop_rows[0]["stop_id"], "name": stop_rows[0]["stop_name"]},
        "destination": {"stop_id": stop_rows[-1]["stop_id"], "name": stop_rows[-1]["stop_name"]},
        "stop_count": len(stop_rows),
        "shape_id": shape_id,
        "shape_point_count": len(shape_rows),
        "route_length_km": route_distances[-1],
        "scheduled_trip_count": len(schedule_rows),
        "target_trip_count": len(target_rows),
        "target_first_departure": target_rows[0]["scheduled_origin_departure"],
        "target_last_departure": target_rows[-1]["scheduled_origin_departure"],
        "target_last_arrival": target_rows[-1]["scheduled_destination_arrival"],
        "files_prefix": prefix,
    }
    write_json(output_dir / f"{prefix}_metadata.json", metadata)
    return metadata


def self_test() -> None:
    assert ROUTE_SHORT_NAME == "50X"
    assert gtfs_time_seconds("25:01:02") == 90062
    assert datetime.strptime("20260817", "%Y%m%d").weekday() == 0
    print("UTA 50X static extractor self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    service_date = args.service_date or default_service_date()
    if datetime.strptime(service_date, "%Y%m%d").weekday() >= 5:
        raise ValueError("The requested service date must be a weekday")
    if args.gtfs_zip:
        gtfs_zip, source_url = args.gtfs_zip, str(args.gtfs_zip)
    else:
        gtfs_zip, source_url = download_gtfs()
    metadata = extract(
        gtfs_zip,
        service_date,
        args.output_dir,
        args.headsign_token,
        args.first_departure,
        args.trip_count,
        source_url,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
