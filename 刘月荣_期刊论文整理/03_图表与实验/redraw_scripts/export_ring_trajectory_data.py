from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = (
    ROOT
    / "00_原始材料"
    / "2022级-刘月荣-论文及相关资料"
    / "代码"
    / "a_biyelunwen2_15"
)
OUT_ROOT = ROOT / "03_图表与实验" / "source_data" / "ring_trajectory_raw"
END_TIME = 28_800

STRATEGIES = {
    "NH": {"project": "base_mappo", "control": 0, "model": None},
    "FH": {"project": "base_mappo", "control": 1, "model": None},
    "BH": {"project": "base_mappo", "control": 3, "model": None},
    "HH": {"project": "base_mappo", "control": 4, "model": None},
    "SH": {"project": "base_mappo", "control": 5, "model": None},
    "MAPPO": {"project": "base_mappo", "control": 2, "model": "mappo"},
    "improved_MAPPO": {
        "project": "base_amappo",
        "control": 2,
        "model": "mappo_caac",
    },
}


def build_args() -> SimpleNamespace:
    return SimpleNamespace(
        speed_type=1,
        seed=1,
        share_scale=1,
        weight=2,
        all=0,
        overtake=0,
        arr_hold=1,
    )


def load_agent(project_dir: Path, model_name: str, engine, seed: int):
    module_name = "model.MAPPO" if model_name == "mappo" else "model.MAPPO_CAAC"
    Agent = importlib.import_module(module_name).Agent
    agents = {}
    for route_id in engine.route_list:
        agent = Agent(
            state_dim=3,
            name="",
            n_stops=len(engine.busstop_list),
            buslist=engine.bus_list,
            seed=seed,
        )
        actor_path = (
            project_dir
            / "model"
            / "save"
            / f"_A_0_1_1_2_{model_name}_1_actor.pth"
        )
        critic_path = (
            project_dir
            / "model"
            / "save"
            / f"_A_0_1_1_2_{model_name}_1_critic.pth"
        )
        agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        if hasattr(agent, "critic") and critic_path.exists():
            agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
        agent.actor.eval()
        if hasattr(agent, "critic"):
            agent.critic.eval()
        agents[route_id] = agent
    return agents


def export_bus(strategy_dir: Path, bus) -> dict[str, object]:
    output = strategy_dir / f"bus_{bus.id}.csv"
    stop_records = list(getattr(bus, "stops_record", []))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_step", "loc_rad", "station_position", "is_stopped", "stop_id"]
        )
        for index, (time_step, loc) in enumerate(zip(bus.time_step, bus.loc)):
            stop_id = stop_records[index] if index < len(stop_records) else -1
            writer.writerow(
                [
                    int(time_step),
                    f"{float(loc):.10f}",
                    f"{float(loc / (2 * np.pi / 12) + 1):.10f}",
                    int(stop_id != -1),
                    stop_id,
                ]
            )
    loc = np.asarray(bus.loc, dtype=float)
    delta = np.diff(loc)
    wrap = delta < -np.pi
    invalid_backward = (delta < -1e-10) & ~wrap
    return {
        "bus_id": int(bus.id),
        "dispatch_time": int(bus.dispatch_time),
        "records": len(bus.loc),
        "time_start": int(bus.time_step[0]),
        "time_end": int(bus.time_step[-1]),
        "completed_circuits": int(np.sum(wrap)),
        "stopped_steps": int(sum(str(value) != "-1" for value in stop_records)),
        "invalid_backward_steps": int(np.sum(invalid_backward)),
        "file": output.name,
    }


def run_strategy(strategy: str) -> None:
    spec = STRATEGIES[strategy]
    project_dir = CODE_ROOT / str(spec["project"])
    sys.path.insert(0, str(project_dir))

    args = build_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    U = importlib.import_module("sim.util")
    Sim_Engine = importlib.import_module("sim.Sim_Engine")

    stop_list, _ = U.getStopList()
    bus_routes = U.getBusRoute(stop_list, args)
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    stop_list_copy = copy.deepcopy(stop_list)
    bus_list_copy = copy.deepcopy(bus_list)

    arrival_rates = [
        1 / 60 / 2,
        1 / 60 / 2,
        1 / 60 / 1.2,
        1 / 60,
        1 / 60,
        1 / 60 * 3,
        1 / 60 * 4,
        1 / 60 * 2,
        1 / 60,
        1 / 60 / 1.5,
        1 / 60 / 1.8,
        1 / 60 / 2,
    ]
    for index, stop in enumerate(stop_list_copy.values()):
        stop.set_rate(arrival_rates[index])

    engine = Sim_Engine.Engine(
        bus_list=bus_list_copy,
        busstop_list=stop_list_copy,
        control_type=int(spec["control"]),
        dispatch_times=dispatch_times,
        demand=0,
        simulation_step=simulation_step,
        route_list=route_list,
        hold_once_arr=args.arr_hold,
        is_allow_overtake=args.overtake,
        share_scale=args.share_scale,
        weight=args.weight,
        all=args.all,
    )
    if spec["model"]:
        engine.agents = load_agent(project_dir, str(spec["model"]), engine, args.seed)

    while engine.simulation_step <= END_TIME:
        engine.sim()

    strategy_dir = OUT_ROOT / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    bus_summaries = [
        export_bus(strategy_dir, bus)
        for bus in sorted(engine.bus_list.values(), key=lambda item: item.id)
    ]
    manifest = {
        "strategy": strategy,
        "project": spec["project"],
        "control_type": spec["control"],
        "model": spec["model"],
        "seed": args.seed,
        "simulation_end_time_s": END_TIME,
        "station_count": len(engine.busstop_list),
        "bus_count": len(engine.bus_list),
        "arrival_rates_per_second": arrival_rates,
        "buses": bus_summaries,
    }
    (strategy_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=STRATEGIES)
    selected = parser.parse_args()
    run_strategy(selected.strategy)


if __name__ == "__main__":
    main()
