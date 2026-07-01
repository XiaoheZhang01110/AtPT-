from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTCollection


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "redrawn_figures"
OUTPUT_PDF = OUTPUT_DIR / "fig_t34_up_route_osm.pdf"
TEMP_PREVIEW = Path("/tmp/fig_t34_up_route_osm_preview.png")
SONGTI_REGULAR = Path("/tmp/SongtiSC-Regular.ttf")

# Geographic operating extent reported for the study-period T34 service.
LON_MIN, LON_MAX = 104.055325, 104.098295
LAT_MIN, LAT_MAX = 30.477385, 30.548793

# Route vertices digitized from the thesis study-period map. Pixel coordinates
# are georeferenced to the reported WGS-84 route extent below.
PIXEL_ROUTE_UP = np.array(
    [
        (868, 171), (868, 190), (887, 206), (878, 218), (852, 218),
        (839, 202), (829, 184), (782, 163), (748, 124), (707, 113),
        (646, 83), (621, 74), (547, 53), (547, 21), (514, 21),
        (514, 170), (426, 170), (414, 188), (414, 271), (393, 271),
        (378, 271), (282, 271), (251, 232), (192, 232), (238, 345),
        (216, 377), (192, 425), (160, 443), (200, 471), (235, 517),
        (282, 525), (282, 615), (356, 616), (382, 636), (366, 662),
        (350, 683),
    ],
    dtype=float,
)


def pixel_to_lonlat(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    x0, x1 = PIXEL_ROUTE_UP[:, 0].min(), PIXEL_ROUTE_UP[:, 0].max()
    y0, y1 = PIXEL_ROUTE_UP[:, 1].min(), PIXEL_ROUTE_UP[:, 1].max()
    lon = LON_MIN + (x - x0) / (x1 - x0) * (LON_MAX - LON_MIN)
    lat = LAT_MAX - (y - y0) / (y1 - y0) * (LAT_MAX - LAT_MIN)
    return np.column_stack([lon, lat])


def interpolate_along_line(points: np.ndarray, count: int) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(0.0, cumulative[-1], count)
    result = []
    for target in targets:
        segment = min(np.searchsorted(cumulative, target, side="right") - 1, len(lengths) - 1)
        segment = max(segment, 0)
        fraction = (target - cumulative[segment]) / lengths[segment] if lengths[segment] else 0.0
        result.append(points[segment] + fraction * (points[segment + 1] - points[segment]))
    return np.asarray(result)


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def fetch_osm_mosaic(bounds: tuple[float, float, float, float], zoom: int = 14):
    lon_min, lat_min, lon_max, lat_max = bounds
    tx0, ty1 = lonlat_to_tile(lon_min, lat_min, zoom)
    tx1, ty0 = lonlat_to_tile(lon_max, lat_max, zoom)
    x0, x1 = math.floor(tx0), math.floor(tx1)
    y0, y1 = math.floor(ty0), math.floor(ty1)
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    headers = {"User-Agent": "academic-route-figure/1.0"}
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            tile = Image.open(io.BytesIO(response.content)).convert("RGB")
            mosaic.paste(tile, ((x - x0) * 256, (y - y0) * 256))
    west, north = tile_to_lonlat(x0, y0, zoom)
    east, south = tile_to_lonlat(x1 + 1, y1 + 1, zoom)
    return mosaic, (west, east, south, north)


def add_scale_bar(ax, lon: float, lat: float, fontproperties: FontProperties, length_km: float = 2.0):
    degree_lon = length_km / (111.32 * math.cos(math.radians(lat)))
    ax.plot([lon, lon + degree_lon], [lat, lat], color="black", lw=2.2, solid_capstyle="butt", zorder=8)
    ax.plot([lon, lon], [lat - 0.0006, lat + 0.0006], color="black", lw=1.2, zorder=8)
    ax.plot([lon + degree_lon, lon + degree_lon], [lat - 0.0006, lat + 0.0006], color="black", lw=1.2, zorder=8)
    ax.text(
        lon + degree_lon / 2,
        lat + 0.0011,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontproperties=fontproperties,
    )


def main() -> None:
    if not SONGTI_REGULAR.exists():
        collection = TTCollection("/System/Library/Fonts/Supplemental/Songti.ttc")
        collection.fonts[6].save(SONGTI_REGULAR)
    songti = FontProperties(fname=SONGTI_REGULAR, weight="normal", size=26)
    mpl.rcParams.update(
        {
            "font.family": "STSong",
            "font.size": 26,
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    route_ll = pixel_to_lonlat(PIXEL_ROUTE_UP)
    stops_ll = interpolate_along_line(route_ll, 39)
    pad_lon, pad_lat = 0.006, 0.006
    bounds = (
        route_ll[:, 0].min() - pad_lon,
        route_ll[:, 1].min() - pad_lat,
        route_ll[:, 0].max() + pad_lon,
        route_ll[:, 1].max() + pad_lat,
    )
    basemap, extent = fetch_osm_mosaic(bounds)

    fig, ax = plt.subplots(figsize=(11.5, 15.2), constrained_layout=True)
    ax.imshow(basemap, extent=extent, origin="upper", alpha=0.72, zorder=0)

    route_color = "#0B6E99"
    accent = "#D1495B"
    ax.plot(
        route_ll[:, 0],
        route_ll[:, 1],
        color="white",
        lw=6.2,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        route_ll[:, 0],
        route_ll[:, 1],
        color=route_color,
        lw=3.6,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=4,
    )

    ax.scatter(
        stops_ll[:, 0],
        stops_ll[:, 1],
        s=30,
        facecolor="white",
        edgecolor=route_color,
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        stops_ll[[0, -1], 0],
        stops_ll[[0, -1], 1],
        s=72,
        facecolor=accent,
        edgecolor="white",
        linewidth=1.5,
        zorder=6,
    )

    ax.annotate(
        "新兴客运站（起点）",
        stops_ll[0],
        xytext=(-12, -48),
        textcoords="offset points",
        ha="right",
        va="top",
        fontproperties=songti,
        bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": accent, "lw": 0.9},
        arrowprops={"arrowstyle": "-", "color": accent, "lw": 1.0},
        zorder=9,
    )
    ax.annotate(
        "香沙路（终点）",
        stops_ll[-1],
        xytext=(10, -16),
        textcoords="offset points",
        ha="left",
        fontproperties=songti,
        bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": accent, "lw": 0.9},
        arrowprops={"arrowstyle": "-", "color": accent, "lw": 1.0},
        zorder=9,
    )

    key_stops = {
        15: ("应龙路中", (-30, 34), "right"),
        17: ("应龙南一路", (10, -15), "left"),
        21: ("会龙大道", (-10, -14), "right"),
        22: ("四河村", (10, 10), "left"),
    }
    key_indices = np.array([sequence - 1 for sequence in key_stops])
    ax.scatter(
        stops_ll[key_indices, 0],
        stops_ll[key_indices, 1],
        s=64,
        marker="D",
        facecolor="#F2B134",
        edgecolor="white",
        linewidth=1.2,
        zorder=8,
    )
    for sequence, (name, offset, alignment) in key_stops.items():
        ax.annotate(
            name,
            stops_ll[sequence - 1],
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontproperties=songti,
            color="#171717",
            bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "#F2B134", "lw": 0.8, "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": "#9A6B00", "lw": 0.8},
            zorder=9,
        )

    # Two arrows make the up-direction unambiguous without a large legend.
    for start_idx in (7, 24):
        p0, p1 = stops_ll[start_idx], stops_ll[start_idx + 1]
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops={"arrowstyle": "-|>", "color": accent, "lw": 1.8, "mutation_scale": 12},
            zorder=8,
        )

    add_scale_bar(ax, bounds[0] + 0.003, bounds[1] + 0.003, songti)
    ax.text(
        0.995,
        0.008,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontproperties=FontProperties(fname=SONGTI_REGULAR, weight="normal", size=16),
        color="#4A4A4A",
        bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
        zorder=10,
    )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1.0 / math.cos(math.radians(np.mean(route_ll[:, 1]))))
    ax.set_axis_off()

    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(TEMP_PREVIEW, dpi=180, bbox_inches="tight", pad_inches=0.03)
    print(OUTPUT_PDF)
    print(TEMP_PREVIEW)


if __name__ == "__main__":
    main()
