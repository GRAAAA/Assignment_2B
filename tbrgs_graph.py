import math
import heapq
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data_processor import get_site_coords


SPEED_LIMIT        = 60.0    # km/h 
CAPACITY_FLOW      = 1500.0  # veh/hr
CAPACITY_SPD       = 32.0    # km/h
INTERSECTION_DELAY = 0.5     # minutes

# Fundamental diagram coefficients: flow = A*speed^2 + B*speed
_A = -CAPACITY_FLOW / (CAPACITY_SPD ** 2)  # -1.4648375
_B = -2 * CAPACITY_SPD * _A                # 93.75

_FREE_FLOW_THRESHOLD = 351.0  # veh/hr below which speed is capped at limit


# Directed edges (from, to, distance_km) for the Boroondara SCATS network
BOROONDARA_EDGES = [
    # Burke Road (N-S)
    ("4030", "4032", 0.76), ("4032", "4034", 0.89),
    ("4034", "4035", 0.64), ("4035", "4040", 1.27),
    ("4040", "4043", 1.55), ("4043", "4273", 1.05),
    ("4032", "4030", 0.76), ("4034", "4032", 0.89),
    ("4035", "4034", 0.64), ("4040", "4035", 1.27),
    ("4043", "4040", 1.55), ("4273", "4043", 1.05),

    # Doncaster / Eastern (E-W)
    ("2827", "4051", 0.92), ("4051", "3180", 0.87),
    ("3180", "2200", 0.93), ("2200", "3126", 1.02),
    ("3126", "0970", 0.94),
    ("4051", "2827", 0.92), ("3180", "4051", 0.87),
    ("2200", "3180", 0.93), ("3126", "2200", 1.02),
    ("0970", "3126", 0.94),

    # Canterbury Road (E-W)
    ("3120", "3122", 0.73), ("3122", "3127", 0.79),
    ("3127", "4063", 0.79), ("4063", "4057", 0.80),
    ("3122", "3120", 0.73), ("3127", "3122", 0.79),
    ("4063", "3127", 0.79), ("4057", "4063", 0.80),

    # Balwyn / High St
    ("3001", "3002", 0.38), ("3002", "3001", 0.38),
    ("3001", "3662", 0.69), ("3662", "3001", 0.69),
    ("3662", "4335", 0.53), ("4335", "3662", 0.53),
    ("4335", "4324", 0.42), ("4324", "4335", 0.42),
    ("4324", "4264", 0.93), ("4264", "4324", 0.93),
    ("4264", "4270", 0.98), ("4270", "4264", 0.98),

    # Burke Rd / Canterbury Rd connectors
    ("4034", "3002", 0.88), ("3002", "4034", 0.88),
    ("4030", "4321", 0.72), ("4321", "4030", 0.72),
    ("4321", "3662", 0.58), ("3662", "4321", 0.58),

    # Warrigal Road (N-S)
    ("2000", "3682", 1.43), ("3682", "3685", 1.77),
    ("3685", "0970", 0.94),
    ("3682", "2000", 1.43), ("3685", "3682", 1.77),
    ("0970", "3685", 0.94),
    ("2000", "3120", 1.43), ("3120", "2000", 1.43),

    # Toorak / Riversdale
    ("2000", "4272", 1.15), ("4272", "2000", 1.15),
    ("4272", "4270", 0.82), ("4270", "4272", 0.82),
    ("4272", "4273", 1.54), ("4273", "4272", 1.54),

    # Glenferrie / Auburn
    ("4263", "4264", 0.89), ("4264", "4263", 0.89),
    ("4262", "4263", 0.89), ("4263", "4262", 0.89),
    ("4270", "4263", 0.73), ("4263", "4270", 0.73),
    ("4812", "4263", 0.52), ("4263", "4812", 0.52),
    ("4821", "4262", 0.72), ("4262", "4821", 0.72),
    ("4821", "3001", 0.70), ("3001", "4821", 0.70),

    # North connections
    ("2825", "4030", 0.83), ("4030", "2825", 0.83),
    ("2820", "3662", 0.52), ("3662", "2820", 0.52),
    ("2825", "2827", 1.10), ("2827", "2825", 1.10),

    # Trafalgar Rd
    ("3804", "3812", 0.52), ("3812", "3804", 0.52),
    ("3804", "4040", 0.66), ("4040", "3804", 0.66),
    ("3812", "4043", 0.62), ("4043", "3812", 0.62),

    # Balwyn Rd
    ("3180", "4057", 0.60), ("4057", "3180", 0.60),

    # Doncaster E of Bulleen
    ("4051", "4030", 0.75), ("4030", "4051", 0.75),
]

BOROONDARA_EDGES = [(f, t, d) for f, t, d in BOROONDARA_EDGES if f != t]


def _load_coords() -> dict:
    try:
        return get_site_coords()
    except Exception:
        return {
            "0970": (-37.86703, 145.09159), "2000": (-37.85168, 145.09435),
            "2200": (-37.81631, 145.09812), "2820": (-37.79477, 145.03077),
            "2825": (-37.78661, 145.06202), "2827": (-37.78093, 145.07733),
            "3001": (-37.81441, 145.02243), "3002": (-37.81489, 145.02663),
            "3120": (-37.82264, 145.05734), "3122": (-37.82379, 145.06466),
            "3126": (-37.82778, 145.09885), "3127": (-37.82506, 145.07800),
            "3180": (-37.79611, 145.08372), "3662": (-37.80876, 145.02757),
            "3682": (-37.83695, 145.09699), "3685": (-37.85467, 145.09384),
            "3804": (-37.83331, 145.06247), "3812": (-37.83738, 145.06119),
            "4030": (-37.79561, 145.06251), "4032": (-37.80202, 145.06127),
            "4034": (-37.81147, 145.05946), "4035": (-37.81727, 145.05836),
            "4040": (-37.83256, 145.05545), "4043": (-37.84683, 145.05275),
            "4051": (-37.79419, 145.06960), "4057": (-37.80431, 145.08197),
            "4063": (-37.81404, 145.08010), "4262": (-37.82155, 145.01503),
            "4263": (-37.82285, 145.02513), "4264": (-37.82389, 145.03409),
            "4270": (-37.82951, 145.03304), "4272": (-37.83186, 145.04668),
            "4273": (-37.84632, 145.04378), "4321": (-37.80078, 145.04946),
            "4324": (-37.80927, 145.03731), "4335": (-37.80624, 145.03518),
            "4812": (-37.82859, 145.01644), "4821": (-37.81285, 145.00849),
        }


_COORDS = None


def get_coords() -> dict:
    global _COORDS
    if _COORDS is None:
        _COORDS = _load_coords()
    return _COORDS


def flow_to_speed(flow_per_15min: float) -> float:
    """Convert predicted flow (veh/15min) to speed (km/h) via the fundamental diagram."""
    flow     = max(0.0, flow_per_15min * 4.0)
    disc     = _B ** 2 + 4 * _A * flow
    if disc < 0:
        return max(1.0, CAPACITY_SPD)
    sqrt_d   = math.sqrt(disc)
    speed_hi = (-_B - sqrt_d) / (2 * _A)   # free-flow branch
    speed_lo = (-_B + sqrt_d) / (2 * _A)   # congested branch
    if flow <= _FREE_FLOW_THRESHOLD:
        return min(max(speed_hi, 1.0), SPEED_LIMIT)
    elif flow <= CAPACITY_FLOW:
        return max(1.0, min(speed_hi, SPEED_LIMIT))
    return max(1.0, speed_lo)


def travel_time_minutes(flow_per_15min: float, distance_km: float) -> float:
    speed = flow_to_speed(flow_per_15min)
    return (distance_km / speed) * 60.0 + INTERSECTION_DELAY


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_graph(predictor=None, predict_day: str = "10/16/2006",
                time_slot: int = 32, model: str = "best") -> dict:
    """Build adjacency list with ML-predicted travel-time edge weights."""
    coords = get_coords()
    graph  = {}
    for (from_id, to_id, dist_km) in BOROONDARA_EDGES:
        if from_id not in coords or to_id not in coords:
            continue
        if predictor is not None:
            try:
                flow = predictor.predict(predict_day, time_slot, model)
            except Exception:
                flow = 50.0
        else:
            flow = 50.0
        weight = travel_time_minutes(flow, dist_km)
        graph.setdefault(from_id, []).append((to_id, weight))
    return graph


def _h(node, destinations, coords):
    """Admissible heuristic: straight-line travel time at speed limit."""
    if node not in coords:
        return 0.0
    lat1, lon1 = coords[node]
    best = math.inf
    for dest in destinations:
        if dest not in coords:
            continue
        lat2, lon2 = coords[dest]
        t = (haversine_km(lat1, lon1, lat2, lon2) / SPEED_LIMIT) * 60.0
        if t < best:
            best = t
    return best if best != math.inf else 0.0


def astar_tbrgs(graph: dict, origin: str, destinations: list) -> tuple:
    """A* on the TBRGS weighted graph. Returns (goal, time_min, path, nodes_created)."""
    coords   = get_coords()
    dest_set = set(destinations)
    if origin in dest_set:
        return origin, 0.0, [origin], 1

    g_cost   = {origin: 0.0}
    counter  = 0
    frontier = [(_h(origin, destinations, coords), origin, counter, 0.0, [origin])]
    explored = set()
    nodes_created = 1

    while frontier:
        _, current, _, g_val, path = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        if current in dest_set:
            return current, g_val, path, nodes_created
        for neighbour, edge_cost in sorted(graph.get(current, []), key=lambda e: e[0]):
            if neighbour in explored:
                continue
            tg = g_val + edge_cost
            if neighbour not in g_cost or tg < g_cost[neighbour]:
                g_cost[neighbour] = tg
                counter += 1
                nodes_created += 1
                f_new = tg + _h(neighbour, destinations, coords)
                heapq.heappush(frontier, (f_new, neighbour, counter,
                                          tg, path + [neighbour]))
    return None, math.inf, [], nodes_created


def top_k_paths(graph: dict, origin: str, destination: str, k: int = 5) -> list:
    """
    Return up to k shortest travel-time paths from origin to destination.
    Uses iterative edge-blocking (Yen-style spur approach).
    """
    results       = []
    candidates    = []
    blocked_edges = set()

    def _run(g):
        return astar_tbrgs(g, origin, [destination])

    goal, t, path, nc = _run(graph)
    if goal is None:
        return []
    results.append({"rank": 1, "path": path, "time_min": round(t, 2),
                    "nodes_created": nc})

    for rank in range(2, k + 1):
        prev_path = results[-1]["path"]
        for spur_idx in range(len(prev_path) - 1):
            spur_to   = prev_path[spur_idx + 1]
            block_key = (prev_path[spur_idx], spur_to)
            mod_graph = {
                node: [(nb, w) for nb, w in nbrs
                       if (node, nb) != block_key
                       and (node, nb) not in blocked_edges]
                for node, nbrs in graph.items()
            }
            g2, t2, p2, nc2 = _run(mod_graph)
            if g2 is not None and tuple(p2) not in {tuple(r["path"]) for r in results}:
                candidates.append((t2, p2, nc2))
            blocked_edges.add(block_key)

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        t_b, p_b, nc_b = candidates.pop(0)
        results.append({"rank": rank, "path": p_b,
                        "time_min": round(t_b, 2), "nodes_created": nc_b})

    return results


if __name__ == "__main__":
    print("Building graph (default flow)...")
    graph = build_graph()
    print(f"  Nodes: {len(graph)}")
    print("\nTop-5 paths: 2000 -> 3002")
    for r in top_k_paths(graph, "2000", "3002", k=5):
        print(f"  Route {r['rank']}: {' -> '.join(r['path'])}  "
              f"({r['time_min']:.1f} min)")
