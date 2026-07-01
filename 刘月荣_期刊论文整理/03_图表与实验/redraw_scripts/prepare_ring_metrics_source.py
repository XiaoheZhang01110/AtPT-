import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MATERIAL_ROOT = ROOT.parent / "00_原始材料" / "2022级-刘月荣-论文及相关资料"
FIGURE_DIR = MATERIAL_ROOT / "学位论文-终稿" / "figures"
NH_LOG = (
    MATERIAL_ROOT
    / "代码"
    / "a_biyelunwen2_15"
    / "base_line"
    / "log"
    / "A_0_1_0_2all_1res.csv"
)
OUTPUT = ROOT / "source_data" / "ring_metrics" / "ring_metrics_by_stop.csv"

METHOD_COLORS = {
    "FH": (94, 128, 183),
    "BH": (102, 176, 119),
    "HH": (224, 144, 99),
    "SH": (201, 95, 99),
    "MAPPO": (141, 128, 186),
    "improved_MAPPO": (95, 212, 228),
}

# Pixel-to-data calibration from the original figures' labeled y-axis ticks.
METRICS = {
    "AWT": {"file": "AWT.png", "scale": 400 / 76, "nh_column": "stw"},
    "AHT": {"file": "AHT.png", "scale": 30 / 79, "nh_column": "sth"},
    "ATT": {"file": "ATT.png", "scale": 800 / 81, "nh_column": "att"},
    "AOD": {"file": "AOD.png", "scale": 6 / 83, "nh_column": "sto"},
}


def x_centers(mask: np.ndarray) -> list[int]:
    xs = np.where(mask.sum(axis=0) > 1)[0]
    xs = xs[xs < 1400]
    groups: list[list[int]] = []
    for x in xs:
        if not groups or x > groups[-1][-1] + 1:
            groups.append([int(x)])
        else:
            groups[-1].append(int(x))
    return [(group[0] + group[-1]) // 2 for group in groups[:12]]


def digitize_metric(path: Path, scale: float) -> dict[str, np.ndarray]:
    image = np.asarray(Image.open(path).convert("RGB"))
    masks = {
        method: np.all(image == np.asarray(color), axis=2)
        for method, color in METHOD_COLORS.items()
    }
    first_bar_centers = x_centers(masks["FH"])
    second_bar_centers = x_centers(masks["SH"])
    result: dict[str, np.ndarray] = {}
    for method, mask in masks.items():
        centers = first_bar_centers if method in {"FH", "BH", "HH"} else second_bar_centers
        result[method] = np.asarray([mask[:, x].sum() * scale for x in centers])
    return result


def main() -> None:
    nh_data = pd.read_csv(NH_LOG).tail(12).reset_index(drop=True)
    records = []
    method_order = ["NH", "HH", "SH", "FH", "BH", "MAPPO", "improved_MAPPO"]

    for metric, metadata in METRICS.items():
        digitized = digitize_metric(FIGURE_DIR / metadata["file"], metadata["scale"])
        values = {"NH": nh_data[metadata["nh_column"]].to_numpy(dtype=float), **digitized}
        for method in method_order:
            for stop, value in enumerate(values[method], start=1):
                records.append(
                    {
                        "metric": metric,
                        "stop": stop,
                        "method": method,
                        "value": round(float(value), 3),
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
