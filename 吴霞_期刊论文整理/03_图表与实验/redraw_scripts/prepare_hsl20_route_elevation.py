#!/usr/bin/env python3
"""Prepare elevation and OSM signal-node data for the HSL 20 figure.

The script uses the official HSL GTFS shape as the route reference, samples
terrain from the Open-Meteo elevation endpoint, and map-matches OpenStreetMap
``highway=traffic_signals`` nodes to a 50 m corridor around the route. Nearby
OSM signal nodes are clustered along the route so a multi-head intersection is
represented by one operational signal location.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "03_图表与实验" / "source_data" / "hsl20_realtime"
PREFIX = "hsl20_eira_to_munkkivuori_20260814"
SHAPE_PATH = DATA_DIR / f"{PREFIX}_shape.csv"
ELEVATION_PATH = DATA_DIR / f"{PREFIX}_elevation.csv"
SIGNALS_PATH = DATA_DIR / f"{PREFIX}_osm_signals.csv"
METADATA_PATH = DATA_DIR / f"{PREFIX}_route_context_metadata.json"

ELEVATION_ENDPOINT = "https://api.open-meteo.com/v1/elevation"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
USER_AGENT = "HSL20AcademicFigure/1.0 (route-elevation visualization)"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def request_json(url: str, data: bytes | None = None, timeout: int = 90) -> dict:
    request = Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_elevations(latitudes: list[float], longitudes: list[float]) -> list[float]:
    elevations: list[float] = []
    for start in range(0, len(latitudes), 100):
        lat_chunk = latitudes[start : start + 100]
        lon_chunk = longitudes[start : start + 100]
        query = urlencode(
            {
                "latitude": ",".join(f"{value:.6f}" for value in lat_chunk),
                "longitude": ",".join(f"{value:.6f}" for value in lon_chunk),
            }
        )
        payload = request_json(f"{ELEVATION_ENDPOINT}?{query}")
        values = payload.get("elevation")
        if not isinstance(values, list) or len(values) != len(lat_chunk):
            raise RuntimeError("Elevation response does not match the requested coordinates")
        elevations.extend(float(value) for value in values)
    return elevations


def local_xy(latitude: float, longitude: float, reference_latitude: float) -> tuple[float, float]:
    x = longitude * 111_320.0 * math.cos(math.radians(reference_latitude))
    y = latitude * 110_540.0
    return x, y


def project_to_route(
    latitude: float,
    longitude: float,
    route_latitudes: list[float],
    route_longitudes: list[float],
    route_distances_km: list[float],
) -> tuple[float, float]:
    reference_latitude = sum(route_latitudes) / len(route_latitudes)
    px, py = local_xy(latitude, longitude, reference_latitude)
    route_xy = [
        local_xy(lat, lon, reference_latitude)
        for lat, lon in zip(route_latitudes, route_longitudes)
    ]
    best_offset_m = float("inf")
    best_distance_km = 0.0
    for index in range(len(route_xy) - 1):
        ax, ay = route_xy[index]
        bx, by = route_xy[index + 1]
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        fraction = 0.0 if denominator == 0 else ((px - ax) * dx + (py - ay) * dy) / denominator
        fraction = max(0.0, min(1.0, fraction))
        qx, qy = ax + fraction * dx, ay + fraction * dy
        offset_m = math.hypot(px - qx, py - qy)
        if offset_m < best_offset_m:
            best_offset_m = offset_m
            best_distance_km = route_distances_km[index] + fraction * (
                route_distances_km[index + 1] - route_distances_km[index]
            )
    return best_distance_km, best_offset_m


def fetch_osm_signals(
    route_latitudes: list[float],
    route_longitudes: list[float],
    route_distances_km: list[float],
) -> tuple[list[dict[str, object]], int]:
    margin_degrees = 0.008
    south = min(route_latitudes) - margin_degrees
    north = max(route_latitudes) + margin_degrees
    west = min(route_longitudes) - margin_degrees
    east = max(route_longitudes) + margin_degrees
    query = (
        "[out:json][timeout:60];"
        f'node["highway"="traffic_signals"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});'
        "out body;"
    )
    payload = request_json(
        OVERPASS_ENDPOINT,
        data=urlencode({"data": query}).encode("utf-8"),
        timeout=120,
    )
    elements = payload.get("elements", [])
    candidates: list[dict[str, object]] = []
    for element in elements:
        if element.get("type") != "node" or "lat" not in element or "lon" not in element:
            continue
        distance_km, offset_m = project_to_route(
            float(element["lat"]),
            float(element["lon"]),
            route_latitudes,
            route_longitudes,
            route_distances_km,
        )
        if offset_m <= 50.0:
            tags = element.get("tags", {})
            candidates.append(
                {
                    "osm_node_id": int(element["id"]),
                    "latitude": float(element["lat"]),
                    "longitude": float(element["lon"]),
                    "route_distance_km": distance_km,
                    "route_offset_m": offset_m,
                    "crossing": tags.get("crossing", ""),
                    "signal_direction": tags.get("traffic_signals:direction", ""),
                }
            )

    candidates.sort(key=lambda row: (float(row["route_distance_km"]), float(row["route_offset_m"])))
    clusters: list[list[dict[str, object]]] = []
    for candidate in candidates:
        if not clusters or (
            float(candidate["route_distance_km"])
            - float(clusters[-1][0]["route_distance_km"])
        ) > 0.075:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    signals: list[dict[str, object]] = []
    for index, cluster in enumerate(clusters, start=1):
        representative = min(cluster, key=lambda row: float(row["route_offset_m"]))
        signals.append(
            {
                "signal_index": index,
                **representative,
                "osm_nodes_in_cluster": len(cluster),
                "coordinate_method": "OSM node map-matched to HSL GTFS shape",
            }
        )
    return signals, len(elements)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    shape = read_csv(SHAPE_PATH)
    if len(shape) != 257:
        raise ValueError(f"Expected 257 GTFS shape points, found {len(shape)}")

    latitudes = [float(row["shape_pt_lat"]) for row in shape]
    longitudes = [float(row["shape_pt_lon"]) for row in shape]
    distances_km = [float(row["shape_dist_traveled_km"]) for row in shape]
    elevations = fetch_elevations(latitudes, longitudes)
    elevation_rows = [
        {
            "shape_pt_sequence": row["shape_pt_sequence"],
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "route_distance_km": f"{distance:.3f}",
            "elevation_m": f"{elevation:.1f}",
            "source": "Open-Meteo Elevation API",
        }
        for row, latitude, longitude, distance, elevation in zip(
            shape, latitudes, longitudes, distances_km, elevations
        )
    ]
    write_csv(
        ELEVATION_PATH,
        elevation_rows,
        [
            "shape_pt_sequence",
            "latitude",
            "longitude",
            "route_distance_km",
            "elevation_m",
            "source",
        ],
    )

    signals, osm_signal_nodes_in_bbox = fetch_osm_signals(latitudes, longitudes, distances_km)
    write_csv(
        SIGNALS_PATH,
        signals,
        [
            "signal_index",
            "osm_node_id",
            "latitude",
            "longitude",
            "route_distance_km",
            "route_offset_m",
            "crossing",
            "signal_direction",
            "osm_nodes_in_cluster",
            "coordinate_method",
        ],
    )

    metadata = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": "HSL 20 Eira -> Munkkivuori",
        "gtfs_shape_points": len(shape),
        "route_length_km": distances_km[-1],
        "elevation_source": ELEVATION_ENDPOINT,
        "elevation_samples": len(elevations),
        "elevation_min_m": min(elevations),
        "elevation_max_m": max(elevations),
        "osm_source": "OpenStreetMap contributors",
        "osm_license": "ODbL",
        "overpass_endpoint": OVERPASS_ENDPOINT,
        "osm_signal_nodes_in_query_bbox": osm_signal_nodes_in_bbox,
        "route_signal_candidates_after_50m_filter_and_75m_clustering": len(signals),
        "signal_matching_note": (
            "OSM traffic-signal nodes within 50 m of the HSL GTFS shape; adjacent nodes "
            "within 75 m along the route are represented by the closest node."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {ELEVATION_PATH} ({len(elevation_rows)} rows)")
    print(f"Saved {SIGNALS_PATH} ({len(signals)} clustered signal locations)")
    print(f"Saved {METADATA_PATH}")


if __name__ == "__main__":
    main()
