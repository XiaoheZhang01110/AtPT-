#!/usr/bin/env python3
"""Download official UDOT ATSPM phase charts and extract observed green windows."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urljoin, urlencode
from zoneinfo import ZoneInfo

from PIL import Image

from uta50x_common import DATA_DIR, gtfs_time_seconds, read_csv, request_bytes, write_csv, write_json


ATSPM_BASE = "https://udottraffic.utah.gov/ATSPM/"
ATSPM_METRIC_URL = urljoin(ATSPM_BASE, "DefaultCharts/GetTimingAndActuations")
TIMEZONE = ZoneInfo("America/Denver")


class ImageSourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attributes = dict(attrs)
        if attributes.get("src"):
            self.sources.append(str(attributes["src"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect official ATSPM timing for UTA 50X signals.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prefix")
    parser.add_argument("--phases", default="1-8", help="Comma/dash list, default 1-8")
    parser.add_argument("--max-signals", type=int, help="Optional connectivity-test limit")
    parser.add_argument("--window-start-local", help="ISO local override, e.g. 2026-08-14T07:00:00-06:00")
    parser.add_argument("--window-end-local", help="ISO local override")
    parser.add_argument("--request-pause-seconds", type=float, default=0.35)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def detect_prefix(data_dir: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    files = sorted(data_dir.glob("uta50x_wvc_to_murray_*_metadata.json"))
    if not files:
        raise FileNotFoundError("No UTA 50X metadata file found")
    return files[-1].name.removesuffix("_metadata.json")


def parse_number_list(value: str) -> list[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = (int(part) for part in item.split("-", 1))
            result.update(range(min(left, right), max(left, right) + 1))
        else:
            result.add(int(item))
    return sorted(result)


def service_datetime(service_date: str, gtfs_time: str) -> datetime:
    day = date(int(service_date[:4]), int(service_date[4:6]), int(service_date[6:8]))
    return datetime.combine(day, datetime.min.time(), tzinfo=TIMEZONE) + timedelta(
        seconds=gtfs_time_seconds(gtfs_time)
    )


def metric_window(metadata: dict) -> tuple[datetime, datetime]:
    start = service_datetime(metadata["service_date"], metadata["target_first_departure"]) - timedelta(minutes=15)
    end = service_datetime(metadata["service_date"], metadata["target_last_arrival"]) + timedelta(minutes=15)
    return start, end


def atspm_date(value: datetime) -> str:
    return value.strftime("%m/%d/%Y %I:%M %p")


def post_metric(signal_id: str, phase: int, start: datetime, end: datetime) -> tuple[str, list[str]]:
    body = {
        "SignalID": signal_id,
        "MetricTypeID": 17,
        "StartDate": atspm_date(start),
        "EndDate": atspm_date(end),
        "YAxisMin": 0,
        "YAxisMax": None,
        "Y2AxisMin": 0,
        "Y2AxisMax": 0,
        "ShowLegend": False,
        "ShowHeaderForEachPhase": True,
        "CombineLanesForEachGroup": True,
        "DotAndBarSize": 2,
        "PhaseFilter": str(phase),
        "PhaseEventCodes": "",
        "GlobalCustomEventCodes": "",
        "GlobalCustomEventParams": "",
        "ExtendVsdSearch": 0,
        "ShowVehicleSignalDisplay": True,
        "ShowPedestrianIntervals": False,
        "ShowPedestrianActuation": False,
        "ExtendStartStopSearch": 0,
        "ShowStopBarPresence": False,
        "ShowLaneByLaneCount": False,
        "ShowAdvancedDilemmaZone": False,
        "ShowAdvancedCount": False,
        "AdvancedOffset": 0,
        "ShowAllLanesInfo": False,
        "ShowLinesStartEnd": False,
        "ShowEventPairs": False,
        "ShowRawEventData": False,
        "ShowPermissivePhases": True,
    }
    html = request_bytes(
        ATSPM_METRIC_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/html, */*; q=0.01",
            "Origin": "https://udottraffic.utah.gov",
            "Referer": urljoin(ATSPM_BASE, "DefaultCharts"),
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=180,
        attempts=2,
    ).decode("utf-8", errors="replace")
    parser = ImageSourceParser()
    parser.feed(html)
    chart_urls = list(
        dict.fromkeys(
            urljoin(ATSPM_BASE, source)
            for source in parser.sources
            if "close" not in source.lower()
        )
    )
    return html, chart_urls


PALETTES = {
    "green": [(60, 179, 113), (0, 250, 154), (50, 205, 50), (144, 238, 144)],
    "yellow": [(255, 255, 0)],
    "red": [(178, 34, 34), (240, 128, 128)],
    "gray": [(220, 220, 220)],
}


def close_to(pixel: tuple[int, int, int], colors: list[tuple[int, int, int]], tolerance: float = 48.0) -> bool:
    return any(math.dist(pixel, color) <= tolerance for color in colors)


def largest_contiguous(values: list[int], gap: int = 2) -> tuple[int, int] | None:
    if not values:
        return None
    best = current = [values[0], values[0]]
    for value in values[1:]:
        if value - current[1] <= gap:
            current[1] = value
        else:
            if current[1] - current[0] > best[1] - best[0]:
                best = current
            current = [value, value]
    if current[1] - current[0] > best[1] - best[0]:
        best = current
    return best[0], best[1]


def green_windows_from_chart(
    path: Path, start: datetime, end: datetime
) -> tuple[list[tuple[datetime, datetime]], dict[str, object]]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    pixels = image.load()
    valid_columns: list[int] = []
    vertical_grid_columns: list[int] = []
    green_scores: list[int] = [0] * width
    for x in range(width):
        counts = {name: 0 for name in PALETTES}
        dark_count = 0
        for y in range(max(0, int(height * 0.08)), min(height, int(height * 0.90))):
            pixel = pixels[x, y]
            if max(pixel) <= 80:
                dark_count += 1
            for name, colors in PALETTES.items():
                if close_to(pixel, colors):
                    counts[name] += 1
                    break
        if dark_count >= height * 0.20:
            vertical_grid_columns.append(x)
        if sum(counts.values()) >= height * 0.20:
            valid_columns.append(x)
        green_scores[x] = counts["green"]
    grid_bounds = (
        (min(vertical_grid_columns), max(vertical_grid_columns))
        if vertical_grid_columns
        and max(vertical_grid_columns) - min(vertical_grid_columns) >= width * 0.35
        else None
    )
    bounds = grid_bounds or largest_contiguous(valid_columns, gap=3)
    if not bounds or bounds[1] - bounds[0] < width * 0.35:
        return [], {"status": "plot_bounds_not_detected", "image_width": width, "image_height": height}
    x0, x1 = bounds
    threshold = height * 0.06
    green_columns = [x for x in range(x0, x1 + 1) if green_scores[x] >= threshold]
    segments: list[tuple[int, int]] = []
    if green_columns:
        current = [green_columns[0], green_columns[0]]
        for value in green_columns[1:]:
            if value - current[1] <= 3:
                current[1] = value
            else:
                if current[1] - current[0] >= 1:
                    segments.append((current[0], current[1]))
                current = [value, value]
        if current[1] - current[0] >= 1:
            segments.append((current[0], current[1]))
    duration = (end - start).total_seconds()
    windows = [
        (
            start + timedelta(seconds=(left - x0) / (x1 - x0) * duration),
            start + timedelta(seconds=(right - x0) / (x1 - x0) * duration),
        )
        for left, right in segments
    ]
    return windows, {
        "status": "ok",
        "image_width": width,
        "image_height": height,
        "plot_x_min": x0,
        "plot_x_max": x1,
        "seconds_per_pixel": duration / (x1 - x0),
        "green_window_count": len(windows),
    }


def collect(args: argparse.Namespace) -> dict:
    prefix = detect_prefix(args.data_dir, args.prefix)
    metadata = json.loads((args.data_dir / f"{prefix}_metadata.json").read_text(encoding="utf-8"))
    signals = read_csv(args.data_dir / f"{prefix}_udot_signals.csv")
    if args.max_signals is not None:
        signals = signals[: args.max_signals]
    phases = parse_number_list(args.phases)
    start, end = metric_window(metadata)
    if args.window_start_local:
        start = datetime.fromisoformat(args.window_start_local).astimezone(TIMEZONE)
    if args.window_end_local:
        end = datetime.fromisoformat(args.window_end_local).astimezone(TIMEZONE)
    chart_dir = args.data_dir / f"{prefix}_atspm_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for signal in signals:
        signal_id = signal["udot_signal_id"]
        for phase in phases:
            status = ""
            error = ""
            chart_count = 0
            extracted_count = 0
            try:
                html, chart_urls = post_metric(signal_id, phase, start, end)
                html_path = chart_dir / f"signal_{signal_id}_phase_{phase}_response.html"
                html_path.write_text(html, encoding="utf-8")
                chart_count = len(chart_urls)
                if not chart_urls:
                    heading = re.search(r"<h1>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
                    status = "no_chart"
                    error = re.sub(r"<[^>]+>", "", heading.group(1)).strip() if heading else "No chart URL returned"
                for chart_index, chart_url in enumerate(chart_urls, start=1):
                    extension = Path(chart_url.split("?", 1)[0]).suffix or ".jpg"
                    chart_path = chart_dir / f"signal_{signal_id}_phase_{phase}_panel_{chart_index}{extension}"
                    chart_path.write_bytes(request_bytes(chart_url, timeout=180, attempts=2))
                    windows, diagnostics = green_windows_from_chart(chart_path, start, end)
                    if diagnostics["status"] != "ok":
                        error = str(diagnostics["status"])
                    for window_index, (green_start, green_end) in enumerate(windows, start=1):
                        duration_s = (green_end - green_start).total_seconds()
                        if duration_s < 1.0:
                            continue
                        window_rows.append(
                            {
                                "signal_index": signal["signal_index"],
                                "udot_signal_id": signal_id,
                                "route_distance_km": signal["route_distance_km"],
                                "controller_phase": phase,
                                "chart_panel_index": chart_index,
                                "green_window_index": window_index,
                                "green_start_local": green_start.isoformat(),
                                "green_end_local": green_end.isoformat(),
                                "green_duration_s": f"{duration_s:.2f}",
                                "chart_file": chart_path.name,
                                "seconds_per_pixel": f"{diagnostics['seconds_per_pixel']:.4f}",
                                "source": ATSPM_METRIC_URL,
                                "method": "colour segmentation of official ATSPM phase-event chart",
                            }
                        )
                        extracted_count += 1
                if chart_urls and not status:
                    status = "ok" if extracted_count else "chart_without_detectable_green"
            except Exception as exc:
                status = "request_error"
                error = f"{type(exc).__name__}: {exc}"
            status_rows.append(
                {
                    "signal_index": signal["signal_index"],
                    "udot_signal_id": signal_id,
                    "controller_phase": phase,
                    "status": status,
                    "chart_count": chart_count,
                    "extracted_green_window_count": extracted_count,
                    "error": error,
                    "window_start_local": start.isoformat(),
                    "window_end_local": end.isoformat(),
                    "source": ATSPM_METRIC_URL,
                }
            )
            print(json.dumps(status_rows[-1], ensure_ascii=False), flush=True)
            time.sleep(args.request_pause_seconds)

    status_fields = [
        "signal_index",
        "udot_signal_id",
        "controller_phase",
        "status",
        "chart_count",
        "extracted_green_window_count",
        "error",
        "window_start_local",
        "window_end_local",
        "source",
    ]
    window_fields = [
        "signal_index",
        "udot_signal_id",
        "route_distance_km",
        "controller_phase",
        "chart_panel_index",
        "green_window_index",
        "green_start_local",
        "green_end_local",
        "green_duration_s",
        "chart_file",
        "seconds_per_pixel",
        "source",
        "method",
    ]
    write_csv(args.data_dir / f"{prefix}_signal_timing_status.csv", status_rows, status_fields)
    write_csv(args.data_dir / f"{prefix}_official_green_windows_all_phases.csv", window_rows, window_fields)
    successful_signals = {
        row["udot_signal_id"] for row in status_rows if row["status"] in ("ok", "chart_without_detectable_green")
    }
    event_signals = {row["udot_signal_id"] for row in window_rows}
    summary = {
        "prefix": prefix,
        "source": ATSPM_METRIC_URL,
        "window_start_local": start.isoformat(),
        "window_end_local": end.isoformat(),
        "signal_count": len(signals),
        "signals_with_official_charts": len(successful_signals),
        "signals_with_phase_events": len(event_signals),
        "phase_requests": len(status_rows),
        "official_green_window_count": len(window_rows),
        "timing_semantics": "Observed actuated phase intervals, not a fabricated fixed-time plan.",
    }
    write_json(args.data_dir / f"{prefix}_signal_timing_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def self_test() -> None:
    assert parse_number_list("1-3,5") == [1, 2, 3, 5]
    assert largest_contiguous([1, 2, 3, 10, 11]) == (1, 3)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.jpg"
        image = Image.new("RGB", (1000, 300), "white")
        pixels = image.load()
        for x in range(100, 901):
            color = (220, 220, 220)
            if 200 <= x <= 300 or 500 <= x <= 600:
                color = (60, 179, 113)
            for y in range(80, 241):
                pixels[x, y] = color
        image.save(path, quality=100, subsampling=0)
        start = datetime(2026, 8, 17, 6, 0, tzinfo=TIMEZONE)
        windows, diagnostics = green_windows_from_chart(path, start, start + timedelta(hours=1))
        assert diagnostics["status"] == "ok"
        assert len(windows) == 2
    print("UTA 50X ATSPM collector self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
