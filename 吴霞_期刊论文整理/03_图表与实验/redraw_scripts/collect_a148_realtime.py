#!/usr/bin/env python3
"""Collect real-time positions for Seoul autonomous bus route A148.

The official Seoul bus API is a snapshot API rather than a historical archive.
Run this collector during the weekday A148 operating window.  The API key is
read only from ``SEOUL_BUS_API_KEY`` and is never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROUTE_ID = "101000009"
START_ORD = 1
END_ORD = 41
SEOUL_TZ = ZoneInfo("Asia/Seoul")
DEFAULT_API_URL = "http://ws.bus.go.kr/api/rest/buspos/getBusPosByRouteSt"

FIELDS = [
    "api_received_at_utc",
    "api_received_at_seoul",
    "dataTm",
    "vehId",
    "plainNo",
    "routeId",
    "sectOrd",
    "sectionId",
    "sectDist",
    "fullSectDist",
    "stopFlag",
    "lastStnId",
    "tmX",
    "tmY",
    "posX",
    "posY",
    "busType",
    "congetion",
    "islastyn",
    "trnstnid",
    "rtDist",
    "lastStTm",
    "nextStTm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect A148 positions from the official Seoul bus API."
    )
    parser.add_argument("--route-id", default=ROUTE_ID)
    parser.add_argument("--start-ord", type=int, default=START_ORD)
    parser.add_argument("--end-ord", type=int, default=END_ORD)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--duration-minutes", type=float, default=220.0)
    parser.add_argument("--idle-stop-minutes", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data" / "a148_realtime",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Make one request for authentication/connectivity testing.",
    )
    return parser.parse_args()


def text_of(parent: ET.Element, tag: str) -> str:
    node = parent.find(tag)
    return "" if node is None or node.text is None else node.text.strip()


def parse_response(payload: bytes) -> tuple[str, str, list[dict[str, str]]]:
    root = ET.fromstring(payload)
    header_code = text_of(root, ".//headerCd")
    header_message = text_of(root, ".//headerMsg")
    if not header_code:
        header_code = text_of(root, ".//resultCode")
        header_message = text_of(root, ".//resultMsg")

    rows: list[dict[str, str]] = []
    for item in root.findall(".//itemList"):
        rows.append({child.tag: (child.text or "").strip() for child in item})
    return header_code, header_message, rows


def build_url(args: argparse.Namespace, service_key: str) -> str:
    # Preserve a key that is already percent-encoded while safely encoding '+'.
    encoded_key = urllib.parse.quote(service_key.strip(), safe="%")
    params = (
        f"serviceKey={encoded_key}"
        f"&busRouteId={urllib.parse.quote(args.route_id)}"
        f"&startOrd={args.start_ord}"
        f"&endOrd={args.end_ord}"
    )
    return f"{args.api_url}?{params}"


def fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
            "User-Agent": "A148-research-collector/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def append_rows(path: Path, rows: list[dict[str, str]], now_utc: datetime) -> int:
    exists = path.exists() and path.stat().st_size > 0
    now_seoul = now_utc.astimezone(SEOUL_TZ)
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            enriched = {field: row.get(field, "") for field in FIELDS}
            enriched["api_received_at_utc"] = now_utc.isoformat()
            enriched["api_received_at_seoul"] = now_seoul.isoformat()
            writer.writerow(enriched)
    return len(rows)


def append_log(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_runtime(args: argparse.Namespace) -> str:
    if args.interval < 5 and not args.once:
        raise SystemExit("--interval must be at least 5 seconds.")
    key = os.environ.get("SEOUL_BUS_API_KEY", "").strip()
    key_file = Path(
        os.environ.get("SEOUL_BUS_API_KEY_FILE", args.output_dir / ".api_key")
    )
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(
            "No API key was found. Set SEOUL_BUS_API_KEY or place the encoded "
            f"Public Data Portal service key in {key_file}."
        )
    return key


def main() -> int:
    args = parse_args()
    key = validate_runtime(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    service_day = datetime.now(SEOUL_TZ).strftime("%Y%m%d")
    positions_path = args.output_dir / f"a148_raw_positions_{service_day}.csv"
    log_path = args.output_dir / f"a148_collection_log_{service_day}.jsonl"
    url = build_url(args, key)

    started = time.monotonic()
    deadline = started + (60.0 if args.once else args.duration_minutes * 60.0)
    seen_vehicle = False
    reached_final_section = False
    last_vehicle_seen_at: float | None = None
    request_count = 0
    unique_samples: set[tuple[str, str]] = set()

    print(f"Collecting route {args.route_id} into {positions_path}")
    while time.monotonic() < deadline:
        cycle_started = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        status = "ok"
        header_code = ""
        header_message = ""
        new_rows: list[dict[str, str]] = []
        try:
            payload = fetch(url, args.timeout)
            header_code, header_message, rows = parse_response(payload)
            if header_code not in {"", "0", "00"}:
                status = "api_error"
            for row in rows:
                identity = (row.get("vehId", ""), row.get("dataTm", ""))
                if identity not in unique_samples:
                    unique_samples.add(identity)
                    new_rows.append(row)
            if rows:
                seen_vehicle = True
                last_vehicle_seen_at = time.monotonic()
                reached_final_section |= any(
                    int(float(row.get("sectOrd", "0") or 0)) >= args.end_ord - 1
                    for row in rows
                )
        except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
            status = "request_error"
            header_message = f"{type(exc).__name__}: {exc}"

        if new_rows:
            append_rows(positions_path, new_rows, now_utc)
        request_count += 1
        append_log(
            log_path,
            {
                "received_at_utc": now_utc.isoformat(),
                "request_number": request_count,
                "status": status,
                "header_code": header_code,
                "header_message": header_message,
                "vehicles_returned": len(new_rows),
            },
        )
        print(
            f"{now_utc.astimezone(SEOUL_TZ).isoformat()} "
            f"request={request_count} new_samples={len(new_rows)} status={status}",
            flush=True,
        )

        if args.once:
            return 0 if status == "ok" else 2
        if (
            seen_vehicle
            and reached_final_section
            and last_vehicle_seen_at is not None
            and time.monotonic() - last_vehicle_seen_at >= args.idle_stop_minutes * 60.0
        ):
            print("Round trip appears complete; stopping after the configured idle period.")
            break

        sleep_for = max(0.0, args.interval - (time.monotonic() - cycle_started))
        time.sleep(sleep_for)

    return 0


if __name__ == "__main__":
    sys.exit(main())
