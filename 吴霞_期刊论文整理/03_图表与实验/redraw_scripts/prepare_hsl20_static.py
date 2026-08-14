#!/usr/bin/env python3
"""Extract the Eira-to-Munkkivuori direction of HSL route 20 from GTFS."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


GTFS_URL = "https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip"
ROUTE_ID = "1020"
DIRECTION_ID = "0"
TIMEZONE = ZoneInfo("Europe/Helsinki")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract HSL route 20 direction 0 static route data."
    )
    parser.add_argument("--gtfs-zip", type=Path)
    parser.add_argument("--service-date", help="YYYYMMDD; defaults to today in Helsinki")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "03_图表与实验" / "source_data" / "hsl20_realtime",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def rows_from_zip(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        yield from csv.DictReader(text)


def active_service_ids(archive: zipfile.ZipFile, service_date: str) -> set[str]:
    day = datetime.strptime(service_date, "%Y%m%d")
    weekday = day.strftime("%A").lower()
    active = {
        row["service_id"]
        for row in rows_from_zip(archive, "calendar.txt")
        if row["start_date"] <= service_date <= row["end_date"] and row[weekday] == "1"
    }
    for row in rows_from_zip(archive, "calendar_dates.txt"):
        if row["date"] != service_date:
            continue
        if row["exception_type"] == "1":
            active.add(row["service_id"])
        elif row["exception_type"] == "2":
            active.discard(row["service_id"])
    return active


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def download_gtfs() -> Path:
    target = Path(tempfile.gettempdir()) / "hsl_gtfs_latest.zip"
    request = urllib.request.Request(GTFS_URL, headers={"User-Agent": "HSL20-research-collector/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    return target


def extract(gtfs_zip: Path, service_date: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gtfs_zip) as archive:
        routes = [row for row in rows_from_zip(archive, "routes.txt") if row["route_id"] == ROUTE_ID]
        if len(routes) != 1:
            raise ValueError(f"Expected one route {ROUTE_ID}, found {len(routes)}")

        active = active_service_ids(archive, service_date)
        trips = [
            row
            for row in rows_from_zip(archive, "trips.txt")
            if row["route_id"] == ROUTE_ID
            and row["direction_id"] == DIRECTION_ID
            and row["service_id"] in active
        ]
        if not trips:
            raise ValueError(f"No active direction-0 trips on {service_date}")

        trip_by_id = {row["trip_id"]: row for row in trips}
        stop_times_by_trip: dict[str, list[dict]] = {trip_id: [] for trip_id in trip_by_id}
        for row in rows_from_zip(archive, "stop_times.txt"):
            if row["trip_id"] in stop_times_by_trip:
                stop_times_by_trip[row["trip_id"]].append(row)
        for values in stop_times_by_trip.values():
            values.sort(key=lambda row: int(row["stop_sequence"]))

        reference_trip = min(
            trips,
            key=lambda row: stop_times_by_trip[row["trip_id"]][0]["departure_time"],
        )
        reference_times = stop_times_by_trip[reference_trip["trip_id"]]
        stop_ids = {row["stop_id"] for row in reference_times}
        stops = {
            row["stop_id"]: row
            for row in rows_from_zip(archive, "stops.txt")
            if row["stop_id"] in stop_ids
        }

        shape_id = reference_trip["shape_id"]
        shape_rows = [
            row for row in rows_from_zip(archive, "shapes.txt") if row["shape_id"] == shape_id
        ]
        shape_rows.sort(key=lambda row: int(row["shape_pt_sequence"]))

    stop_rows = []
    for item in reference_times:
        stop = stops[item["stop_id"]]
        stop_rows.append(
            {
                "stop_sequence": item["stop_sequence"],
                "stop_id": item["stop_id"],
                "stop_code": stop["stop_code"],
                "stop_name": stop["stop_name"],
                "stop_lat": stop["stop_lat"],
                "stop_lon": stop["stop_lon"],
                "shape_dist_traveled_km": item["shape_dist_traveled"],
                "reference_arrival_time": item["arrival_time"],
                "reference_departure_time": item["departure_time"],
            }
        )

    schedule_rows = []
    for trip in trips:
        times = stop_times_by_trip[trip["trip_id"]]
        if not times:
            continue
        schedule_rows.append(
            {
                "service_date": service_date,
                "trip_id": trip["trip_id"],
                "service_id": trip["service_id"],
                "trip_headsign": trip["trip_headsign"],
                "direction_id": trip["direction_id"],
                "shape_id": trip["shape_id"],
                "departure_eira": times[0]["departure_time"],
                "arrival_munkkivuori": times[-1]["arrival_time"],
            }
        )
    schedule_rows.sort(key=lambda row: row["departure_eira"])

    shape_output = [
        {
            "shape_pt_sequence": row["shape_pt_sequence"],
            "shape_pt_lat": row["shape_pt_lat"],
            "shape_pt_lon": row["shape_pt_lon"],
            "shape_dist_traveled_km": row["shape_dist_traveled"],
        }
        for row in shape_rows
    ]

    prefix = f"hsl20_eira_to_munkkivuori_{service_date}"
    write_csv(
        output_dir / f"{prefix}_stops.csv",
        stop_rows,
        list(stop_rows[0]),
    )
    write_csv(
        output_dir / f"{prefix}_timetable.csv",
        schedule_rows,
        list(schedule_rows[0]),
    )
    write_csv(
        output_dir / f"{prefix}_shape.csv",
        shape_output,
        list(shape_output[0]),
    )

    metadata = {
        "source": GTFS_URL,
        "service_date": service_date,
        "route_id": ROUTE_ID,
        "route_short_name": routes[0]["route_short_name"],
        "route_long_name": routes[0]["route_long_name"],
        "direction_id": int(DIRECTION_ID),
        "origin": {"stop_id": stop_rows[0]["stop_id"], "name": stop_rows[0]["stop_name"]},
        "destination": {
            "stop_id": stop_rows[-1]["stop_id"],
            "name": stop_rows[-1]["stop_name"],
        },
        "stop_count": len(stop_rows),
        "shape_id": shape_id,
        "shape_point_count": len(shape_output),
        "route_length_km": float(shape_output[-1]["shape_dist_traveled_km"]),
        "scheduled_trip_count": len(schedule_rows),
        "first_departure": schedule_rows[0]["departure_eira"],
        "last_departure": schedule_rows[-1]["departure_eira"],
    }
    (output_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def self_test() -> None:
    assert ROUTE_ID == "1020"
    assert DIRECTION_ID == "0"
    assert datetime.strptime("20260814", "%Y%m%d").strftime("%A").lower() == "friday"
    print("HSL 20 static extractor self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    service_date = args.service_date or datetime.now(TIMEZONE).strftime("%Y%m%d")
    gtfs_zip = args.gtfs_zip or download_gtfs()
    metadata = extract(gtfs_zip, service_date, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
