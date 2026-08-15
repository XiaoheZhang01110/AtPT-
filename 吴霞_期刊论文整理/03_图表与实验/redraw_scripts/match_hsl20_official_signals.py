#!/usr/bin/env python3
"""Match HSL 20 OSM signal locations to Helsinki's official WFS register."""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


WFS_ENDPOINT = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
WFS_LAYER = "avoindata:Liikennevalot_piste"
USER_AGENT = "HSL20OfficialSignalMatch/1.0 (academic route-data matching)"
MAX_MATCH_DISTANCE_M = 150.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "03_图表与实验" / "source_data" / "hsl20_realtime"
    prefix = "hsl20_eira_to_munkkivuori_20260814"
    parser = argparse.ArgumentParser(
        description="Match clustered OSM traffic signals to Helsinki official intersections."
    )
    parser.add_argument(
        "--osm-signals",
        type=Path,
        default=data_dir / f"{prefix}_osm_signals.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir,
    )
    parser.add_argument("--prefix", default=prefix)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, value)))


def wfs_url() -> str:
    query = urllib.parse.urlencode(
        {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": WFS_LAYER,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
        }
    )
    return f"{WFS_ENDPOINT}?{query}"


def fetch_official_features() -> dict:
    request = urllib.request.Request(wfs_url(), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("type") != "FeatureCollection" or not payload.get("features"):
        raise RuntimeError("Helsinki WFS returned no official traffic-signal features")
    return payload


def confidence(distance_m: float) -> str:
    if distance_m <= 35.0:
        return "high"
    if distance_m <= 75.0:
        return "medium"
    if distance_m <= MAX_MATCH_DISTANCE_M:
        return "review"
    return "unmatched"


def nearest_official(signal: dict[str, str], features: list[dict]) -> tuple[dict | None, float]:
    latitude = float(signal["latitude"])
    longitude = float(signal["longitude"])
    nearest = None
    nearest_distance = float("inf")
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        distance = haversine_m(latitude, longitude, float(coordinates[1]), float(coordinates[0]))
        if distance < nearest_distance:
            nearest = feature
            nearest_distance = distance
    if nearest_distance > MAX_MATCH_DISTANCE_M:
        return None, nearest_distance
    return nearest, nearest_distance


def match(osm_signals: list[dict[str, str]], feature_collection: dict) -> tuple[list[dict], list[dict]]:
    features = feature_collection["features"]
    matches: list[dict] = []
    for signal in osm_signals:
        feature, distance_m = nearest_official(signal, features)
        properties = (feature or {}).get("properties", {})
        geometry = (feature or {}).get("geometry", {})
        coordinates = geometry.get("coordinates", ["", ""])
        matches.append(
            {
                "osm_signal_index": int(signal["signal_index"]),
                "osm_node_id": signal["osm_node_id"],
                "route_distance_km": float(signal["route_distance_km"]),
                "osm_latitude": float(signal["latitude"]),
                "osm_longitude": float(signal["longitude"]),
                "official_feature_id": (feature or {}).get("id", ""),
                "official_row_id": properties.get("id", ""),
                "official_intersection_number": properties.get("numero", ""),
                "official_intersection_name": properties.get("risteys", ""),
                "official_type": properties.get("tyyppi", ""),
                "official_additional_information": properties.get("lisatiedot", ""),
                "official_data_owner": properties.get("datanomistaja", ""),
                "official_updated_at": properties.get("paivitetty_tietopalveluun", ""),
                "official_latitude": coordinates[1] if feature else "",
                "official_longitude": coordinates[0] if feature else "",
                "match_distance_m": round(distance_m, 2),
                "match_confidence": confidence(distance_m),
            }
        )

    counts = Counter(row["official_feature_id"] for row in matches if row["official_feature_id"])
    for row in matches:
        row["osm_locations_mapped_to_same_official_intersection"] = counts.get(
            row["official_feature_id"], 0
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in matches:
        if row["official_feature_id"]:
            grouped[row["official_feature_id"]].append(row)
    intersections: list[dict] = []
    for sequence, (_, rows) in enumerate(
        sorted(grouped.items(), key=lambda item: min(row["route_distance_km"] for row in item[1])),
        start=1,
    ):
        representative = min(rows, key=lambda row: row["match_distance_m"])
        intersections.append(
            {
                "route_signal_sequence": sequence,
                "official_feature_id": representative["official_feature_id"],
                "official_intersection_number": representative["official_intersection_number"],
                "official_intersection_name": representative["official_intersection_name"],
                "official_type": representative["official_type"],
                "route_distance_km": min(row["route_distance_km"] for row in rows),
                "official_latitude": representative["official_latitude"],
                "official_longitude": representative["official_longitude"],
                "osm_signal_indices": ";".join(str(row["osm_signal_index"]) for row in rows),
                "osm_locations_in_group": len(rows),
                "minimum_match_distance_m": min(row["match_distance_m"] for row in rows),
                "worst_match_confidence": max(
                    (row["match_confidence"] for row in rows),
                    key=lambda value: {"high": 0, "medium": 1, "review": 2, "unmatched": 3}[value],
                ),
            }
        )
    return matches, intersections


def self_test() -> None:
    assert haversine_m(60.0, 24.0, 60.0, 24.0) == 0
    assert confidence(20) == "high"
    assert confidence(60) == "medium"
    assert confidence(100) == "review"
    assert confidence(200) == "unmatched"
    print("HSL 20 official signal matching self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    osm_signals = read_csv(args.osm_signals)
    feature_collection = fetch_official_features()
    matches, intersections = match(osm_signals, feature_collection)

    raw_path = args.output_dir / f"{args.prefix}_helsinki_official_signals.geojson"
    match_path = args.output_dir / f"{args.prefix}_signals_official_match.csv"
    unique_path = args.output_dir / f"{args.prefix}_official_signal_intersections.csv"
    summary_path = args.output_dir / f"{args.prefix}_signals_official_match_summary.json"
    raw_path.write_text(json.dumps(feature_collection, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(match_path, matches, list(matches[0]))
    write_csv(unique_path, intersections, list(intersections[0]))

    confidence_counts = Counter(row["match_confidence"] for row in matches)
    summary = {
        "matched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": WFS_ENDPOINT,
        "layer": WFS_LAYER,
        "source_license": "CC BY 4.0",
        "official_features_downloaded": len(feature_collection["features"]),
        "osm_signal_locations": len(osm_signals),
        "matched_osm_signal_locations": sum(row["match_confidence"] != "unmatched" for row in matches),
        "unique_official_intersections": len(intersections),
        "match_confidence_counts": dict(confidence_counts),
        "maximum_accepted_match_distance_m": MAX_MATCH_DISTANCE_M,
        "maximum_observed_match_distance_m": max(row["match_distance_m"] for row in matches),
        "note": (
            "Multiple OSM signal locations can map to one official intersection. "
            "The official WFS exposes identifiers and names but no signal timing plans."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {match_path}")
    print(f"Saved {unique_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
