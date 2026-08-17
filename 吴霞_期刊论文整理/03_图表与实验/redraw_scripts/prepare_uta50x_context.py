#!/usr/bin/env python3
"""Build UTA 50X elevation samples and official UDOT signal inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from uta50x_common import (
    DATA_DIR,
    interpolate_route,
    project_to_route,
    read_csv,
    request_json,
    write_csv,
    write_json,
)


USGS_SAMPLES_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples"
UDOT_SIGNALS_URL = "https://maps.udot.utah.gov/central/rest/services/TrafficAndSafety/AT_Intersection_Safety_Data/MapServer/1/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official elevation and signal context for UTA 50X.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prefix", help="Static file prefix; auto-detected when omitted")
    parser.add_argument("--elevation-samples", type=int, default=426)
    parser.add_argument("--signal-corridor-m", type=float, default=100.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def detect_prefix(data_dir: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    metadata_files = sorted(data_dir.glob("uta50x_wvc_to_murray_*_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError("No UTA 50X static metadata file found")
    return metadata_files[-1].name.removesuffix("_metadata.json")


def fetch_usgs_elevations(samples: list[tuple[float, float, float]]) -> list[float]:
    values: list[float] = []
    for start in range(0, len(samples), 100):
        chunk = samples[start : start + 100]
        geometry = {
            "points": [[longitude, latitude] for _, latitude, longitude in chunk],
            "spatialReference": {"wkid": 4326},
        }
        parameters = urlencode(
            {
                "geometry": json.dumps(geometry, separators=(",", ":")),
                "geometryType": "esriGeometryMultipoint",
                "returnFirstValueOnly": "true",
                "interpolation": "RSP_BilinearInterpolation",
                "outFields": "*",
                "f": "json",
            }
        )
        payload = request_json(f"{USGS_SAMPLES_URL}?{parameters}", timeout=180)
        if "error" in payload:
            raise RuntimeError(f"USGS 3DEP getSamples error: {payload['error']}")
        response_samples = payload.get("samples", [])
        if len(response_samples) != len(chunk):
            raise RuntimeError(
                f"USGS returned {len(response_samples)} samples for a {len(chunk)}-point request"
            )
        for sample in response_samples:
            raw = sample.get("value")
            if raw in (None, "NoData"):
                raise RuntimeError("USGS returned a NoData elevation on the route")
            values.append(float(raw))
    return values


def property_value(properties: dict, *suffixes: str) -> object:
    for suffix in suffixes:
        for key, value in properties.items():
            if key.upper() == suffix.upper() or key.upper().endswith("." + suffix.upper()):
                if value not in (None, ""):
                    return value
    return ""


def fetch_udot_signals(
    route_points: list[tuple[float, float]],
    route_distances: list[float],
    corridor_m: float,
) -> tuple[list[dict[str, object]], int]:
    latitudes = [point[0] for point in route_points]
    longitudes = [point[1] for point in route_points]
    margin = 0.004
    envelope = ",".join(
        f"{value:.7f}"
        for value in (
            min(longitudes) - margin,
            min(latitudes) - margin,
            max(longitudes) + margin,
            max(latitudes) + margin,
        )
    )
    query = urlencode(
        {
            "where": "1=1",
            "outFields": "sig_pred.SIGNALID,sig_pred.SIG_TYPE,sig_pred.ST_EW,sig_pred.ST_NS,sig_pred.CITY",
            "returnGeometry": "true",
            "geometry": envelope,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    payload = request_json(f"{UDOT_SIGNALS_URL}?{query}", timeout=180)
    if "error" in payload:
        raise RuntimeError(f"UDOT signal inventory error: {payload['error']}")
    features = payload.get("features", [])
    matched: dict[str, dict[str, object]] = {}
    for feature in features:
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) < 2:
            continue
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
        route_distance, offset_m = project_to_route(
            latitude, longitude, route_points, route_distances
        )
        if offset_m > corridor_m:
            continue
        properties = feature.get("properties", {})
        signal_id = str(property_value(properties, "SIGNALID"))
        if not signal_id:
            continue
        candidate = {
            "udot_signal_id": signal_id,
            "street_east_west": property_value(properties, "ST_EW"),
            "street_north_south": property_value(properties, "ST_NS"),
            "city": property_value(properties, "CITY"),
            "signal_type": property_value(properties, "SIG_TYPE"),
            "latitude": f"{latitude:.7f}",
            "longitude": f"{longitude:.7f}",
            "route_distance_km": f"{route_distance:.6f}",
            "route_offset_m": f"{offset_m:.2f}",
            "inventory_source": UDOT_SIGNALS_URL,
        }
        previous = matched.get(signal_id)
        if previous is None or float(candidate["route_offset_m"]) < float(previous["route_offset_m"]):
            matched[signal_id] = candidate
    result = sorted(matched.values(), key=lambda row: float(row["route_distance_km"]))
    for index, row in enumerate(result, start=1):
        row["signal_index"] = index
    return result, len(features)


def main() -> int:
    args = parse_args()
    if args.self_test:
        assert property_value({"sig_pred.SIGNALID": 123}, "SIGNALID") == 123
        print("UTA 50X context extractor self-test passed.")
        return 0
    prefix = detect_prefix(args.data_dir, args.prefix)
    shape_path = args.data_dir / f"{prefix}_shape.csv"
    shape = read_csv(shape_path)
    route_points = [(float(row["latitude"]), float(row["longitude"])) for row in shape]
    route_distances = [float(row["route_distance_km"]) for row in shape]

    samples = interpolate_route(route_points, route_distances, args.elevation_samples)
    elevations = fetch_usgs_elevations(samples)
    elevation_rows = [
        {
            "sample_index": index,
            "route_distance_km": f"{distance:.6f}",
            "latitude": f"{latitude:.7f}",
            "longitude": f"{longitude:.7f}",
            "elevation_m": f"{elevation:.3f}",
            "source": "USGS The National Map 3DEP",
        }
        for index, ((distance, latitude, longitude), elevation) in enumerate(
            zip(samples, elevations), start=1
        )
    ]
    write_csv(
        args.data_dir / f"{prefix}_elevation_426.csv",
        elevation_rows,
        list(elevation_rows[0]),
    )

    signals, bbox_feature_count = fetch_udot_signals(
        route_points, route_distances, args.signal_corridor_m
    )
    signal_fields = [
        "signal_index",
        "udot_signal_id",
        "street_east_west",
        "street_north_south",
        "city",
        "signal_type",
        "latitude",
        "longitude",
        "route_distance_km",
        "route_offset_m",
        "inventory_source",
    ]
    write_csv(args.data_dir / f"{prefix}_udot_signals.csv", signals, signal_fields)
    metadata = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "prefix": prefix,
        "elevation_source": USGS_SAMPLES_URL,
        "elevation_sample_count": len(elevation_rows),
        "signal_inventory_source": UDOT_SIGNALS_URL,
        "signal_bbox_feature_count": bbox_feature_count,
        "matched_signal_count": len(signals),
        "signal_corridor_m": args.signal_corridor_m,
        "signal_note": "Official UDOT inventory points map-matched to the official UTA GTFS shape.",
    }
    write_json(args.data_dir / f"{prefix}_context_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
