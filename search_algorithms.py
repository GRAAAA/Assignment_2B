import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

from astar import astar
from bfs   import bfs
from dfs   import dfs
from gbfs  import gbfs
from cus1  import cus1
from cus2  import cus2

from subgraph import SUBGRAPH_COORDS, SUBGRAPH_EDGES, build_subgraph

ALGORITHMS = {
    "astar": astar,
    "bfs":   bfs,
    "dfs":   dfs,
    "gbfs":  gbfs,
    "cus1":  cus1,
    "cus2":  cus2,
}

ALGORITHM_LABELS = {
    "astar": "A* (Informed, Optimal)",
    "bfs":   "BFS (Uninformed, Hop-optimal)",
    "dfs":   "DFS (Uninformed, Not optimal)",
    "gbfs":  "GBFS (Informed, Greedy)",
    "cus1":  "IDDFS / CUS1 (Uninformed, Hop-optimal)",
    "cus2":  "IDA* / CUS2 (Informed, Optimal)",
}

# Boroondara centroid reference for lat/lon -> km conversion
_LAT_KM = 111.0
_LON_KM = 111.0 * math.cos(math.radians(-37.82))


def _latlon_to_xy(lat: float, lon: float) -> tuple:
    ref_lat, ref_lon = -37.82, 145.06
    return (lon - ref_lon) * _LON_KM, (lat - ref_lat) * _LAT_KM


def _build_id_maps(nodes: list) -> tuple:
    sorted_nodes = sorted(nodes)
    str_to_int   = {n: i for i, n in enumerate(sorted_nodes)}
    int_to_str   = {i: n for i, n in enumerate(sorted_nodes)}
    return str_to_int, int_to_str


def _convert_graph(str_graph: dict, str_to_int: dict) -> dict:
    int_graph = {}
    for str_node, neighbours in str_graph.items():
        if str_node not in str_to_int:
            continue
        int_node = str_to_int[str_node]
        int_graph[int_node] = [
            (str_to_int[nb], cost)
            for nb, cost in neighbours
            if nb in str_to_int
        ]
    return int_graph


def _convert_coords(str_to_int: dict) -> dict:
    int_coords = {}
    for str_id, int_id in str_to_int.items():
        if str_id not in SUBGRAPH_COORDS:
            continue
        lat, lon = SUBGRAPH_COORDS[str_id][:2]
        int_coords[int_id] = _latlon_to_xy(lat, lon)
    return int_coords


def run_algorithm(algorithm: str, origin: str, destination: str,
                  predictor=None, predict_day: str = "10/16/2006",
                  time_slot: int = 32, model: str = "best") -> dict:
    """Run one A2A algorithm on the TBRGS subgraph and return a result dict."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. "
                         f"Choose from: {list(ALGORITHMS.keys())}")

    str_graph = build_subgraph(predictor, predict_day, time_slot, model)

    all_nodes          = list(SUBGRAPH_COORDS.keys())
    str_to_int, int_to_str = _build_id_maps(all_nodes)
    int_graph          = _convert_graph(str_graph, str_to_int)
    int_coords         = _convert_coords(str_to_int)

    if origin not in str_to_int:
        raise ValueError(f"Origin '{origin}' not in subgraph. "
                         f"Valid: {sorted(str_to_int.keys())}")
    if destination not in str_to_int:
        raise ValueError(f"Destination '{destination}' not in subgraph. "
                         f"Valid: {sorted(str_to_int.keys())}")

    int_origin = str_to_int[origin]
    int_dest   = [str_to_int[destination]]

    goal_int, nodes_created, int_path = ALGORITHMS[algorithm](
        int_graph, int_coords, int_origin, int_dest)

    str_path = [int_to_str[n] for n in int_path] if int_path else []
    goal_str = int_to_str.get(goal_int) if goal_int is not None else None

    total_time = 0.0
    for i in range(len(str_path) - 1):
        f, t = str_path[i], str_path[i + 1]
        total_time += next((w for nb, w in str_graph.get(f, []) if nb == t), 0.0)

    return {
        "algorithm":      algorithm,
        "label":          ALGORITHM_LABELS[algorithm],
        "origin":         origin,
        "destination":    destination,
        "goal":           goal_str,
        "path":           str_path,
        "nodes_created":  nodes_created,
        "total_time_min": round(total_time, 3),
        "hops":           len(str_path) - 1 if str_path else 0,
        "found":          goal_str is not None,
    }


def run_all_algorithms(origin: str, destination: str,
                       predictor=None, predict_day: str = "10/16/2006",
                       time_slot: int = 32, model: str = "best") -> list:
    """Run all six algorithms and return results sorted by total_time_min."""
    results = []
    for algo in ALGORITHMS:
        try:
            results.append(run_algorithm(algo, origin, destination,
                                         predictor, predict_day, time_slot, model))
        except Exception as e:
            results.append({
                "algorithm": algo, "label": ALGORITHM_LABELS[algo],
                "origin": origin, "destination": destination,
                "goal": None, "path": [], "nodes_created": 0,
                "total_time_min": float("inf"), "hops": 0,
                "found": False, "error": str(e),
            })
    return sorted(results, key=lambda r: r["total_time_min"])


def print_comparison(results: list) -> None:
    print("\n" + "=" * 75)
    print(f"  ALGORITHM COMPARISON: {results[0]['origin']} -> {results[0]['destination']}")
    print("=" * 75)
    print(f"  {'Algorithm':<30} {'Found':>6} {'Time(min)':>10} {'Hops':>6} {'Nodes':>7}")
    print(f"  {'─'*30} {'─'*6} {'─'*10} {'─'*6} {'─'*7}")
    for r in results:
        found  = "Y" if r["found"] else "N"
        time_s = f"{r['total_time_min']:.2f}" if r["found"] else "—"
        hops_s = str(r["hops"]) if r["found"] else "—"
        print(f"  {r['label']:<30} {found:>6} {time_s:>10} {hops_s:>6} "
              f"{r['nodes_created']:>7}")
    print("=" * 75)
    found = [r for r in results if r["found"]]
    if found:
        best = min(found, key=lambda r: r["total_time_min"])
        print(f"  Optimal ({best['algorithm'].upper()}): {' -> '.join(best['path'])}")
        print(f"  Travel time: {best['total_time_min']:.2f} min  "
              f"Hops: {best['hops']}  Nodes: {best['nodes_created']}")
    print()


if __name__ == "__main__":
    origin = sys.argv[1] if len(sys.argv) > 1 else "4030"
    dest   = sys.argv[2] if len(sys.argv) > 2 else "4043"

    print(f"Running all 6 algorithms: {origin} -> {dest}\n")
    results = run_all_algorithms(origin, dest)
    print_comparison(results)
    print("Individual paths:")
    for r in results:
        path_str = " -> ".join(r["path"]) if r["path"] else "No path"
        print(f"  [{r['algorithm'].upper():<5}] {path_str}")
