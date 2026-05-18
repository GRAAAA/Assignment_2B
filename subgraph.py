import sys
import os
import math
import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# 15 SCATS intersections in Boroondara forming three road corridors.
# Real lat/lon from the SCATS dataset.
SUBGRAPH_COORDS = {
    # Burke Road (N to S) — 7 nodes
    "4030": (-37.79561, 145.06251, "BURKE_RD\n/ DONCASTER_RD"),
    "4032": (-37.80202, 145.06127, "BURKE_RD\n/ HARP_RD"),
    "4034": (-37.81147, 145.05946, "BURKE_RD\n/ WHITEHORSE_RD"),
    "4035": (-37.81727, 145.05836, "BURKE_RD\n/ MONT_ALBERT_RD"),
    "3120": (-37.82264, 145.05734, "BURKE_RD\n/ CANTERBURY_RD"),
    "4040": (-37.83256, 145.05545, "BURKE_RD\n/ RIVERSDALE_RD"),
    "4043": (-37.84683, 145.05275, "BURKE_RD\n/ TOORAK_RD"),
    # Canterbury Road branch (east of 3120)
    "3122": (-37.82379, 145.06466, "CANTERBURY_RD\n/ STANHOPE_GV"),
    "3127": (-37.82506, 145.07800, "BALWYN_RD\n/ CANTERBURY_RD"),
    # Balwyn Road (north from 3127)
    "4063": (-37.81404, 145.08010, "BALWYN_RD\n/ WHITEHORSE_RD"),
    "4057": (-37.80431, 145.08197, "BALWYN_RD\n/ BELMORE_RD"),
    "3180": (-37.79611, 145.08372, "DONCASTER_RD\n/ BALWYN_RD"),
    # Riversdale / Tooronga
    "3804": (-37.83331, 145.06247, "TRAFALGAR_RD\n/ RIVERSDALE_RD"),
    "4272": (-37.83186, 145.04668, "RIVERSDALE_RD\n/ TOORONGA_RD"),
    "4273": (-37.84632, 145.04378, "TOORONGA_RD\n/ TOORAK_RD"),
    # Assignment example nodes — referenced in spec as O=2000, D=3002
    "2000": (-37.85168, 145.09435, "WARRIGAL_RD\n/ TOORAK_RD"),
    "3002": (-37.81489, 145.02663, "BARKERS_RD\n/ DENMARK_ST"),
}

_UNDIRECTED_EDGES = [
    # Burke Rd — consecutive intersections, no skipping
    ("4030", "4032", 0.72), ("4032", "4034", 1.06), ("4034", "4035", 0.65),
    ("4035", "3120", 0.60), ("3120", "4040", 1.12), ("4040", "4043", 1.60),
    # Canterbury Rd branch
    ("3120", "3122", 0.66), ("3122", "3127", 1.18),
    # Balwyn Rd branch
    ("3127", "4063", 1.24), ("4063", "4057", 1.09), ("4057", "3180", 0.93),
    # Riversdale / Tooronga
    ("4040", "3804", 0.62), ("4040", "4272", 0.77),
    ("4272", "4273", 1.63), ("4273", "4043", 0.79),
    # Assignment example nodes (O=2000, D=3002 per spec)
    ("2000", "3120", 1.43), ("2000", "4272", 1.15),
    ("3002", "4034", 0.88),
]

# Expand undirected → directed (bidirectional)
SUBGRAPH_EDGES = []
for f, t, d in _UNDIRECTED_EDGES:
    SUBGRAPH_EDGES.append((f, t, d))
    SUBGRAPH_EDGES.append((t, f, d))

SUBGRAPH_ADJ = {}
for f, t, d in SUBGRAPH_EDGES:
    SUBGRAPH_ADJ.setdefault(f, []).append((t, d))

# Maps each directed edge to its real SCATS "Location" string for per-edge RF prediction
EDGE_TO_LOCATION = {
    ("4030", "4032"): "BURKE_RD S of DONCASTER_RD",
    ("4032", "4030"): "BURKE_RD N of HARP_RD",
    ("4032", "4034"): "BURKE_RD S of HARP_RD",
    ("4034", "4032"): "BURKE_RD N OF WHITEHORSE_RD",
    ("4034", "4035"): "BURKE_RD S OF WHITEHORSE_RD",
    ("4035", "4034"): "BURKE_RD N of MONT ALBERT_RD",
    ("4035", "3120"): "BURKE_RD S of BARKERS_RD",
    ("3120", "4035"): "BURKE_RD N of CANTERBURY_RD",
    ("3120", "4040"): "BURKE_RD S of CANTERBURY_RD",
    ("4040", "3120"): "BURKE_RD N of RIVERSDALE_RD",
    ("4040", "4043"): "BURKE_RD S of RIVERSDALE_RD",
    ("4043", "4040"): "BURKE_RD N of TOORAK_RD",
    ("3120", "3122"): "CANTERBURY_RD E of BURKE_RD",
    ("3122", "3120"): "CANTERBURY_RD W of STANHOPE_GV",
    ("3122", "3127"): "CANTERBURY_RD E of STANHOPE_GV",
    ("3127", "3122"): "CANTERBURY_RD W of BALWYN_RD",
    ("3127", "4063"): "BALWYN_RD N of CANTERBURY_RD",
    ("4063", "3127"): "BALWYN_RD S OF WHITEHORSE_RD",
    ("4063", "4057"): "BALWYN_RD N OF WHITEHORSE_RD",
    ("4057", "4063"): "BALWYN_RD S OF BELMORE_RD",
    ("4057", "3180"): "BALWYN_RD N OF BELMORE_RD",
    ("3180", "4057"): "BALWYN_RD S of DONCASTER_RD",
    ("4040", "3804"): "RIVERSDALE_RD E of BURKE_RD",
    ("3804", "4040"): "RIVERSDALE_RD W of TRAFALGAR_RD",
    ("4040", "4272"): "RIVERSDALE_RD W of BURKE_RD",
    ("4272", "4040"): "RIVERSDALE_RD E of TOORONGA_RD",
    ("4272", "4273"): "TOORONGA_RD S of RIVERSDALE_RD",
    ("4273", "4272"): "TOORONGA_RD N of TOORAK_RD",
    ("4273", "4043"): "TOORAK_RD E of TOORONGA_RD",
    ("4043", "4273"): "TOORAK_RD W OF BURKE_RD",
    # Assignment example nodes
    ("2000", "3120"): "BURKE_RD N of CANTERBURY_RD",
    ("3120", "2000"): "BURKE_RD S of CANTERBURY_RD",
    ("2000", "4272"): "RIVERSDALE_RD W of BURKE_RD",
    ("4272", "2000"): "TOORONGA_RD N of TOORAK_RD",
    ("3002", "4034"): "BURKE_RD N OF WHITEHORSE_RD",
    ("4034", "3002"): "BURKE_RD S OF WHITEHORSE_RD",
}


# Flow → travel time (same formula as tbrgs_graph.py)
_A = -1500.0 / (32.0 ** 2)
_B = -2.0 * 32.0 * _A
SPEED_LIMIT         = 60.0
CAPACITY_FLOW       = 1500.0
FREE_FLOW_THRESHOLD = 351.0
INTERSECTION_DELAY  = 0.5


def flow_to_speed(flow_per_15min: float) -> float:
    flow     = max(0.0, flow_per_15min * 4.0)
    disc     = _B ** 2 + 4 * _A * flow
    if disc < 0:
        return 32.0
    sqrt_d   = math.sqrt(disc)
    speed_hi = (-_B - sqrt_d) / (2 * _A)
    speed_lo = (-_B + sqrt_d) / (2 * _A)
    if flow <= FREE_FLOW_THRESHOLD:
        return min(max(speed_hi, 1.0), SPEED_LIMIT)
    elif flow <= CAPACITY_FLOW:
        return max(1.0, min(speed_hi, SPEED_LIMIT))
    return max(1.0, speed_lo)


def travel_time(flow_per_15min: float, dist_km: float) -> float:
    return (dist_km / flow_to_speed(flow_per_15min)) * 60.0 + INTERSECTION_DELAY


def build_subgraph(predictor=None, predict_day: str = "10/16/2006",
                   time_slot: int = 32, model: str = "best",
                   per_edge_flow: bool = True) -> dict:
    graph = {}
    for f, t, dist_km in SUBGRAPH_EDGES:
        if predictor is not None:
            try:
                if per_edge_flow and (f, t) in EDGE_TO_LOCATION:
                    flow = predictor.predict(predict_day, time_slot, model,
                                             location_name=EDGE_TO_LOCATION[(f, t)])
                else:
                    flow = predictor.predict(predict_day, time_slot, model)
            except Exception:
                flow = 50.0
        else:
            flow = 50.0
        graph.setdefault(f, []).append((t, round(travel_time(flow, dist_km), 3)))
    return graph


def _haversine(a, b):
    R    = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    x    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0]))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def _heuristic(node, destinations):
    if node not in SUBGRAPH_COORDS:
        return 0.0
    lat1, lon1 = SUBGRAPH_COORDS[node][:2]
    best = math.inf
    for d in destinations:
        if d not in SUBGRAPH_COORDS:
            continue
        t = (_haversine((lat1, lon1), SUBGRAPH_COORDS[d][:2]) / SPEED_LIMIT) * 60.0
        if t < best:
            best = t
    return best if best != math.inf else 0.0


def astar_subgraph(graph, origin, destinations):
    """A* on the subgraph. Returns (goal, total_min, path, nodes_created)."""
    dest_set = set(destinations)
    if origin in dest_set:
        return origin, 0.0, [origin], 1
    g_cost   = {origin: 0.0}
    counter  = 0
    frontier = [(_heuristic(origin, destinations), origin, counter, 0.0, [origin])]
    explored = set()
    nodes_created = 1
    while frontier:
        _, cur, _, g, path = heapq.heappop(frontier)
        if cur in explored:
            continue
        explored.add(cur)
        if cur in dest_set:
            return cur, g, path, nodes_created
        for nb, w in sorted(graph.get(cur, []), key=lambda e: e[0]):
            if nb in explored:
                continue
            tg = g + w
            if nb not in g_cost or tg < g_cost[nb]:
                g_cost[nb] = tg
                counter += 1
                nodes_created += 1
                f_val = tg + _heuristic(nb, destinations)
                heapq.heappush(frontier, (f_val, nb, counter, tg, path + [nb]))
    return None, math.inf, [], nodes_created


def draw_subgraph(graph=None, highlight_path=None,
                  origin=None, destination=None,
                  title="Boroondara 15-Node SCATS Subgraph",
                  flow_info="", save_path=None, show=False,
                  demo_mode=True):
    coords  = SUBGRAPH_COORDS
    lats    = [v[0] for v in coords.values()]
    lons    = [v[1] for v in coords.values()]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    def proj(nid):
        lat, lon = coords[nid][:2]
        m = 0.09
        x = m + (lon - lon_min) / (lon_max - lon_min + 1e-9) * (1 - 2 * m)
        y = m + (lat - lat_min) / (lat_max - lat_min + 1e-9) * (1 - 2 * m)
        return x, y

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    path_edges = set()
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            path_edges.add((highlight_path[i], highlight_path[i + 1]))

    drawn = set()
    for f, t, dist in SUBGRAPH_EDGES:
        x1, y1  = proj(f)
        x2, y2  = proj(t)
        is_path = (f, t) in path_edges
        pair    = frozenset([f, t])
        color   = "#E65100" if is_path else "#BDBDBD"
        lw      = 2.5 if is_path else 0.9
        alpha   = 1.0 if is_path else 0.6
        dx, dy  = x2 - x1, y2 - y1
        length  = math.sqrt(dx * dx + dy * dy) + 1e-9
        offset  = 0.006 if pair in drawn else 0.0
        px, py  = -dy / length * offset, dx / length * offset
        ax.annotate("", xy=(x2 + px, y2 + py), xytext=(x1 + px, y1 + py),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    alpha=alpha, mutation_scale=13))
        if is_path and pair not in drawn:
            mx, my = (x1 + x2) / 2 + px, (y1 + y2) / 2 + py
            wt = next((w for nb, w in graph.get(f, []) if nb == t), None) if graph else None
            label = (f"{dist:.2f} km" if demo_mode
                     else (f"{dist:.2f}km\n{wt:.1f}min" if wt else f"{dist:.2f}km"))
            ax.text(mx, my, label, fontsize=7.5, color="#E65100",
                    ha="center", va="center", zorder=6, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#E65100", alpha=0.92, lw=0.8))
        drawn.add(pair)

    for nid, info in coords.items():
        x, y    = proj(nid)
        in_path = highlight_path and nid in highlight_path
        if nid == origin:
            color, size, ring = "#2E7D32", 260, 3.5
        elif nid == destination:
            color, size, ring = "#C62828", 260, 3.5
        elif in_path:
            color, size, ring = "#E65100", 180, 2.5
        else:
            color, size, ring = "#1565C0", 130, 1.2
        ax.scatter(x, y, s=size * 1.9, c=color, alpha=0.15, zorder=4)
        ax.scatter(x, y, s=size, c=color, zorder=5,
                   edgecolors="white", linewidths=ring)
        ax.text(x, y, nid, ha="center", va="center", zorder=7,
                fontsize=6.5, fontweight="bold", fontfamily="monospace",
                color="white")
        road = info[2].replace("\n", " / ")
        yoff = 0.046 if y < 0.5 else -0.05
        ax.text(x, y + yoff, road, ha="center", va="center",
                fontsize=6.2, color="#546E7A", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    legend_items = [
        mpatches.Patch(color="#2E7D32", label=f"Origin: {origin or '—'}"),
        mpatches.Patch(color="#C62828", label=f"Destination: {destination or '—'}"),
        mpatches.Patch(color="#E65100", label="Selected path"),
        mpatches.Patch(color="#1565C0", label="SCATS intersection"),
        mpatches.Patch(color="#BDBDBD", label="Road link"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              facecolor="white", edgecolor="#CBD5E0",
              labelcolor="#1A202C", fontsize=8.5, framealpha=0.95)
    ax.set_title(title, color="#1A202C", fontsize=15, fontweight="bold", pad=14)
    if flow_info:
        ax.text(0.5, 1.012, flow_info, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="#546E7A",
                fontfamily="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig, ax


def print_subgraph_summary(graph=None):
    print("=" * 70)
    print("BOROONDARA 15-NODE SCATS SUBGRAPH")
    print("=" * 70)
    print(f"\n{'SCATS ID':<10} {'Lat':>10} {'Lon':>10}  Location")
    print(f"{'─'*10} {'─'*10} {'─'*10}  {'─'*30}")
    for nid, (lat, lon, road) in SUBGRAPH_COORDS.items():
        print(f"{nid:<10} {lat:>10.5f} {lon:>10.5f}  {road.replace(chr(10), ' / ')}")
    print(f"\nNodes: {len(SUBGRAPH_COORDS)}    Directed edges: {len(SUBGRAPH_EDGES)}")
    if graph:
        print(f"\n{'From':<8} {'To':<8} {'Distance':>10}  {'Travel time':>12}")
        print(f"{'─'*8} {'─'*8} {'─'*10}  {'─'*12}")
        shown = set()
        for f, t, dist in SUBGRAPH_EDGES:
            if (f, t) in shown:
                continue
            shown.add((f, t))
            wt   = next((w for nb, w in graph.get(f, []) if nb == t), None)
            wt_s = f"{wt:.2f} min" if wt else "—"
            print(f"{f:<8} {t:<8} {dist:>8.2f} km  {wt_s:>12}")


if __name__ == "__main__":
    origin = sys.argv[1] if len(sys.argv) > 1 else "4030"
    dest   = sys.argv[2] if len(sys.argv) > 2 else "4043"

    graph = build_subgraph()
    print_subgraph_summary(graph)

    print(f"\nA* search: {origin} -> {dest}")
    goal, cost, path, nc = astar_subgraph(graph, origin, [dest])
    if goal:
        print(f"  Path: {' -> '.join(path)}")
        print(f"  Travel time: {cost:.2f} min   Nodes created: {nc}")
        flow_info = f"A*: {origin} -> {dest}  |  {cost:.1f} min  |  {nc} nodes"
    else:
        print("  No path found.")
        path      = None
        flow_info = f"No path: {origin} -> {dest}"

    draw_subgraph(graph=graph, highlight_path=path,
                  origin=origin, destination=dest,
                  flow_info=flow_info,
                  save_path="subgraph_output.png", show=False,
                  demo_mode=True)
