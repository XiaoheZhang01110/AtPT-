#!/usr/bin/env python3
"""Shared geometry, CSV, and HTTP helpers for the UTA 50X dataset pipeline."""

from __future__ import annotations

import csv
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "03_图表与实验" / "source_data" / "uta50x_realtime"
USER_AGENT = "UTA50X-academic-research-collector/1.0"
EARTH_RADIUS_M = 6_371_008.8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_bytes(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    attempts: int = 3,
) -> bytes:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=data, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to fetch {url}: {last_error}") from last_error


def request_json(url: str, **kwargs) -> dict:
    return json.loads(request_bytes(url, **kwargs).decode("utf-8"))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def cumulative_distances_km(points: list[tuple[float, float]]) -> list[float]:
    distances = [0.0]
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        distances.append(distances[-1] + haversine_m(lat1, lon1, lat2, lon2) / 1000.0)
    return distances


def local_xy(latitude: float, longitude: float, reference_latitude: float) -> tuple[float, float]:
    return (
        longitude * 111_320.0 * math.cos(math.radians(reference_latitude)),
        latitude * 110_540.0,
    )


def project_to_route(
    latitude: float,
    longitude: float,
    route_points: list[tuple[float, float]],
    route_distances_km: list[float],
) -> tuple[float, float]:
    """Return distance along route (km) and perpendicular offset (m)."""
    reference_latitude = sum(point[0] for point in route_points) / len(route_points)
    px, py = local_xy(latitude, longitude, reference_latitude)
    route_xy = [local_xy(lat, lon, reference_latitude) for lat, lon in route_points]
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


def interpolate_route(
    route_points: list[tuple[float, float]],
    route_distances_km: list[float],
    sample_count: int,
) -> list[tuple[float, float, float]]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    targets = [route_distances_km[-1] * index / (sample_count - 1) for index in range(sample_count)]
    result: list[tuple[float, float, float]] = []
    segment = 0
    for target in targets:
        while segment + 1 < len(route_distances_km) - 1 and route_distances_km[segment + 1] < target:
            segment += 1
        d0, d1 = route_distances_km[segment], route_distances_km[segment + 1]
        fraction = 0.0 if d1 == d0 else (target - d0) / (d1 - d0)
        lat0, lon0 = route_points[segment]
        lat1, lon1 = route_points[segment + 1]
        result.append((target, lat0 + fraction * (lat1 - lat0), lon0 + fraction * (lon1 - lon0)))
    return result


def gtfs_time_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_gtfs_time(value: int) -> str:
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
