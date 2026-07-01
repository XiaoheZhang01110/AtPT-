from __future__ import annotations

import csv
import time
from pathlib import Path

import requests


STOPS_DOWN = [
    "香沙路",
    "沙河社区",
    "地铁广福站(B口)",
    "两江路天府大道口",
    "华阳客运站",
    "天研路口",
    "南湖路东",
    "二江路一段北",
    "正东中街",
    "正东上街",
    "天府新区人民医院",
    "正北中街华阳大道口",
    "正北中街",
    "华阳中学",
    "正北下街",
    "四河村",
    "会龙大道",
    "会龙大道梓州大道口",
    "新通大道西",
    "应龙南一路南",
    "应龙南一路",
    "应龙北一路口",
    "应龙路中",
    "中柏大道应龙路口",
    "地铁陆肖站",
    "润和路中和二街口",
    "地铁张家寺站",
    "康和路中和二街口",
    "应龙社区",
    "应龙湾",
    "利民学校",
    "沙站",
    "天府新区综合高级中学",
    "凉水井小区",
    "新兴卫生院",
    "新兴小区",
    "新兴客运站",
]

OUT = Path(__file__).resolve().parents[1] / "source_data" / "t34_stops_osm.csv"
URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "academic-route-figure/1.0 (local research use)"}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for up_index, name in enumerate(reversed(STOPS_DOWN), start=1):
        query = f"{name}, 天府新区, 成都, 四川, 中国"
        response = requests.get(
            URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "viewbox": "104.03,30.57,104.13,30.44",
                "bounded": 1,
            },
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            source = results[0].get("display_name", "")
        else:
            lat = ""
            lon = ""
            source = ""
        rows.append(
            {
                "sequence": up_index,
                "stop_name": name,
                "lat": lat,
                "lon": lon,
                "osm_match": source,
            }
        )
        print(up_index, name, lat, lon)
        time.sleep(1.1)

    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
