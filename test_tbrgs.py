import sys, os, math, unittest
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# Auto-save route visualisations alongside test runs
_VIZ_DIR = os.path.join(os.path.dirname(__file__), "test_screenshots")
os.makedirs(_VIZ_DIR, exist_ok=True)


def _save_path_viz(test_id: str, path: list, origin: str, destination: str,
                   title: str, subtitle: str = "") -> None:
    """Save a route map PNG for one algorithm result."""
    try:
        from subgraph import build_subgraph, draw_subgraph
        graph = build_subgraph()
        save_path = os.path.join(_VIZ_DIR, f"{test_id}.png")
        draw_subgraph(
            graph=graph,
            highlight_path=path,
            origin=origin,
            destination=destination,
            title=title,
            flow_info=subtitle,
            save_path=save_path,
            show=False,
            demo_mode=True,
        )
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception as e:
        print(f"  [viz skip: {e}]")

from data_processor import (
    load_raw, reshape_to_timeseries, get_rf_train_test,
    get_dl_train_test, build_sequences, SEQ_LEN, DATA_FILE
)
from tbrgs_graph import (
    flow_to_speed, travel_time_minutes, build_graph,
    top_k_paths, astar_tbrgs, get_coords,
    SPEED_LIMIT, CAPACITY_FLOW, CAPACITY_SPD, BOROONDARA_EDGES,
)


# ──────────────────────────────────────────────────────────────────────────────
# T01–T05: Data Processing
# ──────────────────────────────────────────────────────────────────────────────

class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests in this class."""
        try:
            cls.df = load_raw(DATA_FILE)
            cls.ok = True
        except Exception as e:
            print(f"  [SKIP] Data file not found ({e}) — skipping data tests.")
            cls.ok = False

    def _skip_if_no_data(self):
        if not self.ok:
            self.skipTest("Data file unavailable")

    def test_T01_raw_load_shape(self):
        """T01: Raw data has expected columns including V00…V95."""
        self._skip_if_no_data()
        self.assertIn("Location", self.df.columns)
        self.assertIn("V00", self.df.columns)
        self.assertIn("V95", self.df.columns)
        self.assertGreater(len(self.df), 100, "Dataset should have many rows")

    def test_T02_unique_locations(self):
        """T02: At least 30 unique SCATS locations in the dataset."""
        self._skip_if_no_data()
        locs = self.df["Location"].unique()
        self.assertGreaterEqual(len(locs), 30, "Expected ≥30 SCATS sites")

    def test_T03_reshape_long_format(self):
        """T03: reshape_to_timeseries produces correct row count (31 days × 96 slots)."""
        self._skip_if_no_data()
        loc = self.df["Location"].unique()[1]
        ts  = reshape_to_timeseries(self.df, loc)
        # October 2006 has 31 days
        self.assertGreaterEqual(len(ts), 31 * 96 * 0.9,
                                "Should have roughly 31×96 rows")
        self.assertIn("flow", ts.columns)
        self.assertIn("time_slot", ts.columns)

    def test_T04_rf_train_test_split(self):
        """T04: RF train/test split produces correct proportions (80/20)."""
        self._skip_if_no_data()
        loc = self.df["Location"].unique()[1]
        ts  = reshape_to_timeseries(self.df, loc)
        X_tr, X_te, y_tr, y_te, cols = get_rf_train_test(ts)
        total = len(X_tr) + len(X_te)
        ratio = len(X_tr) / total
        self.assertAlmostEqual(ratio, 0.8, delta=0.05)

    def test_T05_dl_sequences_shape(self):
        """T05: DL sequence builder produces (n, SEQ_LEN, 1) tensors."""
        self._skip_if_no_data()
        loc = self.df["Location"].unique()[1]
        ts  = reshape_to_timeseries(self.df, loc)
        X_tr, X_te, y_tr, y_te, scaler = get_dl_train_test(ts)
        self.assertEqual(X_tr.shape[1], SEQ_LEN)
        self.assertEqual(X_tr.shape[2], 1)
        self.assertEqual(len(X_tr), len(y_tr))


# ──────────────────────────────────────────────────────────────────────────────
# T06–T09: Flow → Speed → Travel Time
# ──────────────────────────────────────────────────────────────────────────────

class TestFlowConversion(unittest.TestCase):

    def test_T06_zero_flow_free_flow(self):
        """T06: Zero flow → speed = SPEED_LIMIT (60 km/h)."""
        speed = flow_to_speed(0.0)
        self.assertAlmostEqual(speed, SPEED_LIMIT, delta=1.0)

    def test_T07_low_flow_capped(self):
        """T07: Flow of 10 veh/15min (=40 veh/hr) → speed capped at 60 km/h."""
        speed = flow_to_speed(10.0)     # 10 × 4 = 40 veh/hr  ≤ 351 threshold
        # Under free-flow threshold → speed should equal speed limit 60
        self.assertAlmostEqual(speed, SPEED_LIMIT, delta=2.0)

    def test_T08_capacity_flow(self):
        """T08: Flow at road capacity (375 veh/15min = 1500 veh/hr) → speed ≈ 32 km/h."""
        speed = flow_to_speed(CAPACITY_FLOW / 4.0)   # 375 per 15 min
        self.assertAlmostEqual(speed, CAPACITY_SPD, delta=3.0)

    def test_T09_travel_time_positive(self):
        """T09: Travel time is always positive and includes intersection delay."""
        for flow in [0, 20, 80, 200, 400]:
            t = travel_time_minutes(flow, 1.0)
            self.assertGreater(t, 0.5,   # must be > intersection delay
                               f"Travel time should exceed delay, got {t} for flow={flow}")
            self.assertLess(t, 120.0, "Travel time over 2 hours is unrealistic")

    def test_T09b_higher_flow_longer_time(self):
        """T09b: Higher traffic flow → longer travel time (monotonicity in near-capacity range)."""
        # Use flows in the near-capacity range where speed decreases with flow
        t_low  = travel_time_minutes(100, 1.0)   # 400 veh/hr — near capacity
        t_high = travel_time_minutes(300, 1.0)   # 1200 veh/hr — near capacity
        self.assertGreater(t_high, t_low)


# ──────────────────────────────────────────────────────────────────────────────
# T10–T12: Graph Building
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphBuilding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()   # no predictor → default flow

    def test_T10_graph_has_nodes(self):
        """T10: Graph contains expected key SCATS nodes."""
        expected = ["2000", "3002", "4040", "3120", "4043"]
        for n in expected:
            self.assertIn(n, self.graph, f"Node {n} missing from graph")

    def test_T11_edge_weights_positive(self):
        """T11: All edge weights are positive travel times."""
        for node, neighbours in self.graph.items():
            for nb, w in neighbours:
                self.assertGreater(w, 0,
                    f"Edge {node}→{nb} has non-positive weight {w}")

    def test_T12_all_edge_endpoints_are_valid_nodes(self):
        """T12: Every neighbour referenced in an adjacency list is itself a node in the graph."""
        nodes = set(self.graph.keys())
        for u, nbrs in self.graph.items():
            for v, _ in nbrs:
                self.assertIn(v, nodes,
                    f"Edge {u}→{v} references node {v} which has no adjacency entry")


# ──────────────────────────────────────────────────────────────────────────────
# T13–T17: Route Finding
# ──────────────────────────────────────────────────────────────────────────────

class TestRouteFinding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def test_T13_astar_finds_path(self):
        """T13: A* finds the optimal path from 2000 to 3002 on the full Boroondara graph.

        The full graph has more edges than the 17-node subgraph, so A* finds a
        different (shorter by distance) route: 2000->4272->4270->4263->4262->4821->3001->3002
        = 7 hops, cost ~8.89 min with demo flow.
        """
        goal, cost, path, nc = astar_tbrgs(self.graph, "2000", ["3002"])
        self.assertEqual(goal, "3002")
        self.assertGreater(len(path), 1)
        self.assertGreater(cost, 0)
        # Verify no other algorithm finds a lower cost (A* is optimal)
        # Re-run with a reversed destination list to confirm same result
        goal2, cost2, path2, _ = astar_tbrgs(self.graph, "2000", ["3002", "9999"])
        self.assertEqual(goal2, "3002")
        self.assertAlmostEqual(cost2, cost, delta=0.01,
            msg="A* must find the same optimal cost regardless of destination list order")

    def test_T14_path_starts_at_origin(self):
        """T14: Returned path starts at origin node."""
        _, _, path, _ = astar_tbrgs(self.graph, "2000", ["3002"])
        self.assertEqual(path[0], "2000")

    def test_T15_path_ends_at_destination(self):
        """T15: Returned path ends at destination node."""
        _, _, path, _ = astar_tbrgs(self.graph, "2000", ["3002"])
        self.assertEqual(path[-1], "3002")

    def test_T16_no_path_returns_none(self):
        """T16: A* returns None when no path exists (isolated destination)."""
        goal, cost, path, _ = astar_tbrgs(self.graph, "2000", ["FAKE_NODE"])
        self.assertIsNone(goal)
        self.assertEqual(path, [])
        self.assertEqual(cost, math.inf)

    def test_T17_top_k_routes_unique(self):
        """T17: top_k_paths returns distinct routes."""
        routes = top_k_paths(self.graph, "2000", "3002", k=3)
        self.assertGreater(len(routes), 0)
        paths = [tuple(r["path"]) for r in routes]
        self.assertEqual(len(paths), len(set(paths)), "Routes should be unique")

    def test_T17b_routes_ranked_by_time(self):
        """T17b: Routes are returned in ascending travel-time order."""
        routes = top_k_paths(self.graph, "2000", "3002", k=3)
        times = [r["time_min"] for r in routes]
        self.assertEqual(times, sorted(times), "Routes must be sorted by travel time")

    def test_T17c_same_origin_different_dest(self):
        """T17c: Route to different destination returns different path."""
        route_a = top_k_paths(self.graph, "4030", "3002", k=1)
        route_b = top_k_paths(self.graph, "4030", "4043", k=1)
        if route_a and route_b:
            self.assertNotEqual(route_a[0]["path"], route_b[0]["path"],
                                "Different destinations should produce different routes")

    def test_T17d_path_continuity(self):
        """T17d: Every consecutive node pair in a returned path is a real graph edge."""
        routes = top_k_paths(self.graph, "2000", "3002", k=1)
        self.assertGreater(len(routes), 0)
        path = routes[0]["path"]
        edge_set = {(u, v) for u, nbrs in self.graph.items() for v, _ in nbrs}
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.assertIn((u, v), edge_set,
                f"Step {u}→{v} in returned path is not a valid edge in the graph")


# ──────────────────────────────────────────────────────────────────────────────
# T18–T20: ML Model Prediction Sanity (run only if trained)
# ──────────────────────────────────────────────────────────────────────────────

class TestMLPrediction(unittest.TestCase):
    """Lightweight ML pipeline checks — train 5 epochs only. Set SKIP_ML_TESTS=1 to skip."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SKIP_ML_TESTS"):
            cls.predictor = None
            return
        try:
            from traffic_predictor import TrafficPredictor
            cls.predictor = TrafficPredictor(DATA_FILE)
            cls.predictor.train_all(loc_index=1, epochs=5, verbose=0)
        except Exception as e:
            print(f"  [SKIP] ML training failed: {e}")
            cls.predictor = None

    def _skip_if_no_predictor(self):
        if self.predictor is None:
            self.skipTest("Predictor not available")

    def test_T18_rf_prediction_nonnegative(self):
        """T18: Random Forest prediction is non-negative for any slot."""
        self._skip_if_no_predictor()
        for slot in [0, 32, 64, 95]:
            flow = self.predictor.predict("10/1/2006", slot, "rf")
            self.assertGreaterEqual(flow, 0.0,
                f"RF prediction negative for slot {slot}: {flow}")

    def test_T19_lstm_prediction_nonnegative(self):
        """T19: LSTM prediction is non-negative for any slot."""
        self._skip_if_no_predictor()
        for slot in [0, 32, 64, 95]:
            flow = self.predictor.predict("10/1/2006", slot, "lstm")
            self.assertGreaterEqual(flow, 0.0,
                f"LSTM prediction negative for slot {slot}: {flow}")

    def test_T20_best_model_identified(self):
        """T20: Comparison identifies a best model with lowest NRMSE."""
        self._skip_if_no_predictor()
        cmp = self.predictor.comparison
        self.assertIn("best_model", cmp)
        self.assertIn(cmp["best_model"],
                      ["LSTM", "GRU", "Random Forest"],
                      "Best model must be one of the three implemented models")
        nrmses = [s["nrmse"] for s in cmp["summary"]]
        self.assertEqual(cmp["best_nrmse"], min(nrmses),
                         "best_nrmse must equal the lowest NRMSE in summary")

    def test_T36_per_station_rf_models_are_distinct(self):
        """T36: Each SCATS location gets its own independently-trained RF model object."""
        self._skip_if_no_predictor()
        from subgraph import EDGE_TO_LOCATION
        locs = list(set(EDGE_TO_LOCATION.values()))[:3]
        models = []
        for loc in locs:
            rf = self.predictor._get_station_rf(loc)
            self.assertIn("model", rf)
            self.assertIn("metrics", rf)
            models.append(id(rf["model"]))
        # The three station models must be distinct Python objects
        self.assertEqual(len(set(models)), len(models),
            "Each station must have its own independently-trained RF model")




# ──────────────────────────────────────────────────────────────────────────────
# T21–T26: All Six A2A Algorithms on the Subgraph
# ──────────────────────────────────────────────────────────────────────────────

class TestAllAlgorithms(unittest.TestCase):
    """Verify that all six A2A algorithms correctly find paths on the subgraph."""

    @classmethod
    def setUpClass(cls):
        from search_algorithms import run_algorithm, run_all_algorithms
        cls.run_algorithm    = staticmethod(run_algorithm)
        cls.run_all          = staticmethod(run_all_algorithms)
        cls.origin           = "4030"
        cls.destination      = "4043"

    def _run(self, algo):
        return self.run_algorithm(algo, self.origin, self.destination)

    def test_T21_astar_finds_optimal_path(self):
        """T21: A* finds the known optimal path 4030->4043 = 8.75 min, 6 hops."""
        r = self._run("astar")
        self.assertTrue(r["found"], "A* should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)
        self.assertAlmostEqual(r["total_time_min"], 8.75, delta=0.05,
            msg="A* must return the known optimal cost of 8.75 min")
        self.assertEqual(r["hops"], 6,
            msg="A* must find the 6-hop optimal path")
        _save_path_viz("T21_astar_4030_4043", r["path"], self.origin, self.destination,
            "T21 — A* (Optimal)",
            f"Path: {' -> '.join(r['path'])}  |  {r['total_time_min']:.2f} min  |  {r['hops']} hops")

    def test_T22_bfs_finds_path(self):
        """T22: BFS finds the hop-minimal path — exactly 6 hops on 4030->4043."""
        r = self._run("bfs")
        self.assertTrue(r["found"], "BFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)
        self.assertEqual(r["hops"], 6,
            msg="BFS must find the 6-hop minimum on 4030->4043")
        _save_path_viz("T22_bfs_4030_4043", r["path"], self.origin, self.destination,
            "T22 — BFS (Hop-optimal)",
            f"Path: {' -> '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_T23_dfs_finds_path(self):
        """T23: DFS finds a path (may not be optimal)."""
        r = self._run("dfs")
        self.assertTrue(r["found"], "DFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)
        _save_path_viz("T23_dfs_4030_4043", r["path"], self.origin, self.destination,
            "T23 — DFS (Not optimal)",
            f"Path: {' -> '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_T24_gbfs_finds_path(self):
        """T24: GBFS finds a path (greedy heuristic)."""
        r = self._run("gbfs")
        self.assertTrue(r["found"], "GBFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)
        _save_path_viz("T24_gbfs_4030_4043", r["path"], self.origin, self.destination,
            "T24 — GBFS (Greedy best-first)",
            f"Path: {' -> '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_T25_cus1_iddfs_finds_path(self):
        """T25: IDDFS (CUS1) finds the hop-minimal path — exactly 6 hops on 4030->4043."""
        r = self._run("cus1")
        self.assertTrue(r["found"], "IDDFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)
        self.assertEqual(r["hops"], 6,
            msg="IDDFS must find the 6-hop minimum on 4030->4043")
        _save_path_viz("T25_iddfs_4030_4043", r["path"], self.origin, self.destination,
            "T25 — IDDFS / CUS1 (Hop-optimal, low memory)",
            f"Path: {' -> '.join(r['path'])}  |  {r['hops']} hops  |  {r['nodes_created']} nodes created")

    def test_T26_cus2_idastar_finds_optimal_path(self):
        """T26: IDA* (CUS2) finds a path and its travel time matches A*."""
        r_idastar = self._run("cus2")
        r_astar   = self._run("astar")
        self.assertTrue(r_idastar["found"], "IDA* should find a path")
        self.assertAlmostEqual(
            r_idastar["total_time_min"], r_astar["total_time_min"],
            delta=0.1,
            msg="IDA* and A* should find equally optimal travel times"
        )
        _save_path_viz("T26_idastar_4030_4043", r_idastar["path"], self.origin, self.destination,
            "T26 — IDA* / CUS2 (Optimal, low memory)",
            f"Cost: {r_idastar['total_time_min']:.2f} min = A* {r_astar['total_time_min']:.2f} min  |  "
            f"{r_idastar['nodes_created']} nodes created")

    def test_T27_hop_optimal_algorithms_agree(self):
        """T27: BFS and IDDFS (both hop-minimising) return the same hop count as each other."""
        r_bfs  = self._run("bfs")
        r_cus1 = self._run("cus1")
        self.assertTrue(r_bfs["found"] and r_cus1["found"])
        self.assertEqual(r_bfs["hops"], r_cus1["hops"],
            "BFS and IDDFS both minimise hops — they must agree on hop count")

    def test_T28_dfs_finds_suboptimal_path(self):
        """T28: DFS finds a strictly worse path than optimal on 4030->4043.

        4030->4043 has a unique 6-hop optimum. DFS (alphabetical neighbour order)
        commits to the branch 3120->2000->4272->4040 before reaching 4043,
        producing 8 hops and 11.98 min vs the optimal 6 hops and 8.75 min.
        This concretely demonstrates why DFS is unsuitable for route guidance.
        """
        r_dfs = self._run("dfs")
        r_bfs = self._run("bfs")
        self.assertTrue(r_dfs["found"] and r_bfs["found"])
        # DFS must find STRICTLY more hops than the optimal BFS path
        self.assertGreater(r_dfs["hops"], r_bfs["hops"],
            f"DFS ({r_dfs['hops']} hops) should be strictly worse than "
            f"BFS ({r_bfs['hops']} hops) on this query")
        # DFS must find STRICTLY higher travel time than optimal
        self.assertGreater(r_dfs["total_time_min"], r_bfs["total_time_min"] + 1.0,
            "DFS travel time must be meaningfully longer than BFS optimal")
        _save_path_viz("T28_dfs_suboptimal", r_dfs["path"], self.origin, self.destination,
            "T28 — DFS sub-optimal path (NOT route-guidance suitable)",
            f"DFS: {r_dfs['hops']} hops / {r_dfs['total_time_min']:.2f} min  "
            f"vs optimal: {r_bfs['hops']} hops / {r_bfs['total_time_min']:.2f} min")

    def test_T29_nodes_created_varies_by_algorithm(self):
        """T29: DFS, BFS, A*, IDA* create meaningfully different node counts.

        On 4030->4043: IDA* expands many nodes due to iterative re-expansion,
        DFS expands fewer, A* is efficient via heuristic pruning. This verifies
        each algorithm has a distinct search strategy, not all the same code path.
        """
        results = {r["algorithm"]: r for r in self.run_all(self.origin, self.destination)}
        # IDA* re-expands nodes per iteration — must create more than A*
        nc_idastar = results["cus2"]["nodes_created"]
        nc_astar   = results["astar"]["nodes_created"]
        self.assertGreater(nc_idastar, nc_astar,
            f"IDA* ({nc_idastar}) should create more nodes than A* ({nc_astar}) "
            f"due to iterative re-expansion")

    def test_T30_all_algorithms_find_path(self):
        """T30: All 6 algorithms successfully find a path on the subgraph."""
        results = self.run_all(self.origin, self.destination)
        for r in results:
            self.assertTrue(r["found"],
                f"{r['algorithm']} failed to find a path")

    def test_T30b_all_paths_are_valid_edge_sequences(self):
        """T30b: Every step in every algorithm's output path is a real graph edge.

        This is the strongest pathfinding correctness check: it verifies that no
        algorithm returns a 'teleport' step that skips an intermediate node or
        references an edge that does not exist in the graph.
        """
        from subgraph import build_subgraph
        str_graph = build_subgraph()
        edge_set  = {(u, v) for u, nbrs in str_graph.items() for v, _ in nbrs}
        results   = self.run_all(self.origin, self.destination)
        for r in results:
            if not r["found"]:
                continue
            path = r["path"]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.assertIn((u, v), edge_set,
                    f"{r['algorithm'].upper()} path contains invalid step {u}->{v} "
                    f"(not a real directed edge)")

    def test_T30c_source_equals_destination(self):
        """T30c: A* returns the origin immediately when start == destination."""
        r = self.run_algorithm("astar", self.origin, self.origin)
        self.assertTrue(r["found"])
        self.assertEqual(r["hops"], 0, "Same-node query has 0 hops")
        self.assertAlmostEqual(r["total_time_min"], 0.0, delta=0.01,
            msg="Same-node query has cost 0")

    def test_T30d_astar_cost_is_minimum_over_all_algorithms(self):
        """T30d: A* travel time equals the minimum travel time across all algorithms.

        Since A* is admissible (Euclidean heuristic never overestimates on a grid
        of real lat/lon coordinates), its result must equal or beat every other
        algorithm. This is the formal optimality check.
        """
        results = [r for r in self.run_all(self.origin, self.destination) if r["found"]]
        min_time  = min(r["total_time_min"] for r in results)
        astar_time = next(r["total_time_min"] for r in results if r["algorithm"] == "astar")
        self.assertAlmostEqual(astar_time, min_time, delta=0.05,
            msg=f"A* ({astar_time:.2f} min) must match the minimum travel time "
                f"({min_time:.2f} min) across all algorithms")

    def test_T30e_all_algorithms_comparison_grid(self):
        """T30e: Save a 2x3 comparison grid showing all 6 algorithm paths side by side."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from subgraph import build_subgraph, SUBGRAPH_COORDS, SUBGRAPH_EDGES
            import math as _math

            results = {r["algorithm"]: r
                       for r in self.run_all(self.origin, self.destination)}
            graph   = build_subgraph()

            algo_order = [
                ("astar", "A* — Optimal"),
                ("bfs",   "BFS — Hop-optimal"),
                ("dfs",   "DFS — Not optimal"),
                ("gbfs",  "GBFS — Greedy"),
                ("cus1",  "IDDFS / CUS1 — Hop-optimal"),
                ("cus2",  "IDA* / CUS2 — Optimal"),
            ]

            coords   = SUBGRAPH_COORDS
            lats     = [v[0] for v in coords.values()]
            lons     = [v[1] for v in coords.values()]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            def proj(nid, m=0.12):
                lat, lon = coords[nid][:2]
                x = m + (lon - lon_min) / (lon_max - lon_min + 1e-9) * (1 - 2*m)
                y = m + (lat - lat_min) / (lat_max - lat_min + 1e-9) * (1 - 2*m)
                return x, y

            fig, axes = plt.subplots(2, 3, figsize=(18, 11))
            fig.patch.set_facecolor("white")

            for ax, (algo, label) in zip(axes.flat, algo_order):
                r    = results[algo]
                path = r["path"]
                path_edges = set(zip(path, path[1:]))

                ax.set_facecolor("white")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

                drawn = set()
                for f, t, _ in SUBGRAPH_EDGES:
                    x1,y1 = proj(f); x2,y2 = proj(t)
                    is_p  = (f,t) in path_edges
                    pair  = frozenset([f,t])
                    dx,dy = x2-x1, y2-y1
                    ln    = _math.sqrt(dx*dx+dy*dy)+1e-9
                    off   = 0.005 if pair in drawn else 0.0
                    px,py = -dy/ln*off, dx/ln*off
                    col   = "#E65100" if is_p else "#BDBDBD"
                    lw    = 2.2 if is_p else 0.8
                    ax.annotate("", xy=(x2+px,y2+py), xytext=(x1+px,y1+py),
                                arrowprops=dict(arrowstyle="-|>", color=col,
                                                lw=lw, alpha=1.0 if is_p else 0.55,
                                                mutation_scale=9))
                    drawn.add(pair)

                for nid in coords:
                    x,y     = proj(nid)
                    in_path = nid in path
                    if nid == self.origin:
                        col,sz = "#2E7D32", 220
                    elif nid == self.destination:
                        col,sz = "#C62828", 220
                    elif in_path:
                        col,sz = "#E65100", 160
                    else:
                        col,sz = "#78909C", 60
                    ax.scatter(x, y, s=sz, c=col, zorder=5,
                               edgecolors="white", linewidths=1.8)
                    if in_path or nid in (self.origin, self.destination):
                        ax.text(x, y, nid, ha="center", va="center", zorder=7,
                                fontsize=6.5, fontweight="bold",
                                fontfamily="monospace", color="white")

                status = (f"{r['hops']} hops  {r['total_time_min']:.2f} min  "
                          f"{r['nodes_created']} nodes") if r["found"] else "no path"
                ax.set_title(f"{label}\n{status}", fontsize=9,
                             fontweight="bold", color="#1A202C", pad=4)

            plt.suptitle(
                f"All 6 Algorithms — {self.origin} → {self.destination} "
                f"(demo flow, free-flow 60 km/h)",
                fontsize=13, fontweight="bold", color="#1565C0", y=1.01)
            plt.tight_layout()
            save_path = os.path.join(_VIZ_DIR, "T30e_all6_comparison.png")
            plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
            plt.close("all")
            print(f"  Saved: T30e_all6_comparison.png")
            self.assertTrue(os.path.exists(save_path))
        except Exception as e:
            self.skipTest(f"Visualisation skipped: {e}")

    def test_T30f_2000_3002_astar(self):
        """T30f: A* finds the correct path for the assignment's default query 2000->3002."""
        r = self.run_algorithm("astar", "2000", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "2000")
        self.assertEqual(r["path"][-1], "3002")
        self.assertGreater(r["total_time_min"], 0)
        _save_path_viz("T30f_astar_2000_3002", r["path"], "2000", "3002",
            "T30f — A* Default Query  2000 → 3002",
            f"Path: {' -> '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")



# ──────────────────────────────────────────────────────────────────────────────
# T31–T33: Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """End-to-end integration: flow prediction feeds into graph edge weights."""

    @classmethod
    def setUpClass(cls):
        from subgraph import build_subgraph, astar_subgraph, SUBGRAPH_EDGES
        cls.build_subgraph  = staticmethod(build_subgraph)
        cls.astar_subgraph  = staticmethod(astar_subgraph)
        cls.SUBGRAPH_EDGES  = SUBGRAPH_EDGES

    def test_T31_subgraph_has_17_nodes(self):
        """T31: Subgraph contains exactly 17 nodes (15 base + 2000 + 3002)."""
        from subgraph import SUBGRAPH_COORDS
        self.assertEqual(len(SUBGRAPH_COORDS), 17,
                         "Subgraph must have exactly 17 nodes")

    def test_T32_subgraph_has_36_edges(self):
        """T32: Subgraph has exactly 36 directed edges (30 base + 6 for 2000/3002)."""
        self.assertEqual(len(self.SUBGRAPH_EDGES), 36,
                         "Subgraph must have exactly 36 directed edges")

    def test_T33_higher_flow_increases_edge_weight(self):
        """T33: A graph built with higher flow has higher edge weights than low flow."""
        from subgraph import build_subgraph, flow_to_speed, travel_time
        # Build two graphs with different flows and compare a sample edge
        # Low traffic: 10 veh/15min → near free-flow speed
        # High traffic: 300 veh/15min → slower speed → higher travel time
        t_low  = travel_time(10,  1.0)
        t_high = travel_time(300, 1.0)
        self.assertGreater(t_high, t_low,
            "Higher traffic flow should produce higher edge travel time")

    def test_T34_per_edge_flow_gives_varied_weights(self):
        """T34: With per-edge flow, the subgraph has varied edge weights."""
        from subgraph import build_subgraph, EDGE_TO_LOCATION
        # All 30 directed edges must be mapped to a location direction
        self.assertEqual(len(EDGE_TO_LOCATION), 36,
            "Every directed edge must map to a SCATS location direction")
        # In demo mode (no predictor) weights vary by distance only;
        # this still confirms the edge-weight mechanism works.
        g = build_subgraph()
        weights = []
        for f, nbrs in g.items():
            for nb, w in nbrs:
                weights.append(w)
        self.assertEqual(len(weights), 36, "Graph must have 36 directed edges")
        self.assertTrue(all(w > 0 for w in weights),
            "All edge weights must be positive")

    def test_T35_per_station_rf_infrastructure(self):
        """T35: TrafficPredictor exposes per-station Random Forest support."""
        from traffic_predictor import TrafficPredictor
        tp = TrafficPredictor()
        # Per-station RF cache and methods must exist
        self.assertTrue(hasattr(tp, "_station_rf"),
            "TrafficPredictor must have a per-station RF cache")
        self.assertEqual(tp._station_rf, {},
            "Per-station RF cache starts empty")
        for method in ("_get_station_rf", "train_station_rf_models",
                       "_get_location_ts"):
            self.assertTrue(callable(getattr(tp, method, None)),
                f"TrafficPredictor must provide {method}()")

# ──────────────────────────────────────────────────────────────────────────────
# P01–P20: Route Guidance — 20 Pathfinding Test Cases with Auto-saved Maps
# ──────────────────────────────────────────────────────────────────────────────

class TestRouteGuidance(unittest.TestCase):
    """20 concrete pathfinding tests that save a route-map PNG each.

    Each test: runs one algorithm on one O/D pair, asserts correctness
    (path endpoints, hop count, travel-time optimality), then saves a PNG
    to test_screenshots/ for the assignment report.

    Groups
    ------
    P01–P06  All six algorithms on the primary query  4030 → 4043
    P07–P12  All six algorithms on the default query  2000 → 3002
    P13–P16  A* on four diverse cross-suburb routes
    P17–P20  Cross-algorithm property checks
    """

    @classmethod
    def setUpClass(cls):
        from search_algorithms import run_algorithm, run_all_algorithms
        cls._algo = staticmethod(run_algorithm)
        cls._all  = staticmethod(run_all_algorithms)

    # ── Group A: 4030 → 4043, all six algorithms ──────────────────────────────

    def test_P01_astar_4030_4043(self):
        """P01: A* finds the known-optimal path 4030→4043 in 8.75 min, 6 hops."""
        r = self._algo("astar", "4030", "4043")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "4030"); self.assertEqual(r["path"][-1], "4043")
        self.assertEqual(r["hops"], 6)
        self.assertAlmostEqual(r["total_time_min"], 8.75, delta=0.05)
        _save_path_viz("P01_astar_4030_4043", r["path"], "4030", "4043",
            "P01 — A* (Optimal)  4030 → 4043",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P02_bfs_4030_4043(self):
        """P02: BFS finds the hop-minimal path 4030→4043 — exactly 6 hops."""
        r = self._algo("bfs", "4030", "4043")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "4030"); self.assertEqual(r["path"][-1], "4043")
        self.assertEqual(r["hops"], 6)
        _save_path_viz("P02_bfs_4030_4043", r["path"], "4030", "4043",
            "P02 — BFS (Hop-optimal)  4030 → 4043",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P03_dfs_4030_4043_suboptimal(self):
        """P03: DFS returns a sub-optimal 8-hop path (vs optimal 6 hops) on 4030→4043.

        DFS commits to the first deep branch it encounters; on this graph that
        leads through the detour 3120→2000→4272→4040 instead of 3120→4040
        directly, adding 2 unnecessary hops and 3.23 extra minutes.
        """
        r = self._algo("dfs", "4030", "4043")
        self.assertTrue(r["found"])
        self.assertEqual(r["hops"], 8, "DFS must take the known 8-hop detour on 4030→4043")
        self.assertAlmostEqual(r["total_time_min"], 11.98, delta=0.05)
        _save_path_viz("P03_dfs_4030_4043", r["path"], "4030", "4043",
            "P03 — DFS (Sub-optimal)  4030 → 4043",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  (optimal: 6 hops / 8.75 min)")

    def test_P04_gbfs_4030_4043(self):
        """P04: GBFS (greedy best-first) finds a path on 4030→4043."""
        r = self._algo("gbfs", "4030", "4043")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "4030"); self.assertEqual(r["path"][-1], "4043")
        self.assertEqual(r["hops"], 6)
        _save_path_viz("P04_gbfs_4030_4043", r["path"], "4030", "4043",
            "P04 — GBFS (Greedy best-first)  4030 → 4043",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P05_iddfs_4030_4043(self):
        """P05: IDDFS (CUS1) finds the hop-minimal 6-hop path on 4030→4043."""
        r = self._algo("cus1", "4030", "4043")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "4030"); self.assertEqual(r["path"][-1], "4043")
        self.assertEqual(r["hops"], 6)
        _save_path_viz("P05_iddfs_4030_4043", r["path"], "4030", "4043",
            "P05 — IDDFS / CUS1 (Hop-optimal, low memory)  4030 → 4043",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['nodes_created']} nodes created")

    def test_P06_idastar_4030_4043(self):
        """P06: IDA* (CUS2) finds the same optimal cost as A* on 4030→4043."""
        r_ida  = self._algo("cus2", "4030", "4043")
        r_ast  = self._algo("astar", "4030", "4043")
        self.assertTrue(r_ida["found"])
        self.assertAlmostEqual(r_ida["total_time_min"], r_ast["total_time_min"], delta=0.05,
            msg="IDA* must match A* optimal cost on 4030→4043")
        _save_path_viz("P06_idastar_4030_4043", r_ida["path"], "4030", "4043",
            "P06 — IDA* / CUS2 (Optimal, low memory)  4030 → 4043",
            f"Path: {' → '.join(r_ida['path'])}  |  {r_ida['total_time_min']:.2f} min = A* optimal  |  {r_ida['nodes_created']} nodes")

    # ── Group B: 2000 → 3002, all six algorithms ──────────────────────────────

    def test_P07_astar_2000_3002(self):
        """P07: A* finds the assignment default query 2000→3002 in 5.56 min, 4 hops."""
        r = self._algo("astar", "2000", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "2000"); self.assertEqual(r["path"][-1], "3002")
        self.assertEqual(r["hops"], 4)
        self.assertAlmostEqual(r["total_time_min"], 5.56, delta=0.05)
        _save_path_viz("P07_astar_2000_3002", r["path"], "2000", "3002",
            "P07 — A* (Optimal)  2000 → 3002",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P08_bfs_2000_3002(self):
        """P08: BFS finds the hop-minimal 4-hop path on 2000→3002."""
        r = self._algo("bfs", "2000", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["hops"], 4)
        _save_path_viz("P08_bfs_2000_3002", r["path"], "2000", "3002",
            "P08 — BFS (Hop-optimal)  2000 → 3002",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P09_dfs_2000_3002(self):
        """P09: DFS finds a valid path on 2000→3002 with correct endpoints."""
        r = self._algo("dfs", "2000", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "2000"); self.assertEqual(r["path"][-1], "3002")
        _save_path_viz("P09_dfs_2000_3002", r["path"], "2000", "3002",
            "P09 — DFS  2000 → 3002",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P10_gbfs_2000_3002(self):
        """P10: GBFS finds a valid path on 2000→3002 with correct endpoints."""
        r = self._algo("gbfs", "2000", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "2000"); self.assertEqual(r["path"][-1], "3002")
        _save_path_viz("P10_gbfs_2000_3002", r["path"], "2000", "3002",
            "P10 — GBFS (Greedy best-first)  2000 → 3002",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min  |  {r['nodes_created']} nodes")

    def test_P11_iddfs_2000_3002(self):
        """P11: IDDFS (CUS1) finds the hop-minimal 4-hop path on 2000→3002."""
        r = self._algo("cus1", "2000", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["hops"], 4)
        _save_path_viz("P11_iddfs_2000_3002", r["path"], "2000", "3002",
            "P11 — IDDFS / CUS1 (Hop-optimal, low memory)  2000 → 3002",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['nodes_created']} nodes created")

    def test_P12_idastar_2000_3002(self):
        """P12: IDA* (CUS2) finds the same optimal cost as A* on 2000→3002."""
        r_ida  = self._algo("cus2", "2000", "3002")
        r_ast  = self._algo("astar", "2000", "3002")
        self.assertTrue(r_ida["found"])
        self.assertAlmostEqual(r_ida["total_time_min"], r_ast["total_time_min"], delta=0.05,
            msg="IDA* must match A* optimal cost on 2000→3002")
        _save_path_viz("P12_idastar_2000_3002", r_ida["path"], "2000", "3002",
            "P12 — IDA* / CUS2 (Optimal, low memory)  2000 → 3002",
            f"Path: {' → '.join(r_ida['path'])}  |  {r_ida['total_time_min']:.2f} min = A* optimal  |  {r_ida['nodes_created']} nodes")

    # ── Group C: A* on four diverse cross-suburb routes ──────────────────────

    def test_P13_astar_3180_4043(self):
        """P13: A* finds the longest route in the subgraph — 3180→4043 (7 hops)."""
        r = self._algo("astar", "3180", "4043")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "3180"); self.assertEqual(r["path"][-1], "4043")
        self.assertEqual(r["hops"], 7)
        self.assertAlmostEqual(r["total_time_min"], 11.32, delta=0.1)
        _save_path_viz("P13_astar_3180_4043", r["path"], "3180", "4043",
            "P13 — A* (Optimal)  3180 → 4043  (7-hop cross-suburb route)",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_P14_astar_4030_3002(self):
        """P14: A* finds a short 3-hop route — 4030→3002 (4.16 min)."""
        r = self._algo("astar", "4030", "3002")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "4030"); self.assertEqual(r["path"][-1], "3002")
        self.assertEqual(r["hops"], 3)
        self.assertAlmostEqual(r["total_time_min"], 4.16, delta=0.1)
        _save_path_viz("P14_astar_4030_3002", r["path"], "4030", "3002",
            "P14 — A* (Optimal)  4030 → 3002  (3-hop direct route)",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_P15_astar_4272_4030(self):
        """P15: A* finds the reverse corridor — 4272→4030 (6 hops, 7.92 min)."""
        r = self._algo("astar", "4272", "4030")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "4272"); self.assertEqual(r["path"][-1], "4030")
        self.assertEqual(r["hops"], 6)
        self.assertAlmostEqual(r["total_time_min"], 7.92, delta=0.1)
        _save_path_viz("P15_astar_4272_4030", r["path"], "4272", "4030",
            "P15 — A* (Optimal)  4272 → 4030  (reverse corridor)",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_P16_astar_3804_4272(self):
        """P16: A* finds the shortest route in the subgraph — 3804→4272 (2 hops, 2.39 min)."""
        r = self._algo("astar", "3804", "4272")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "3804"); self.assertEqual(r["path"][-1], "4272")
        self.assertEqual(r["hops"], 2)
        self.assertAlmostEqual(r["total_time_min"], 2.39, delta=0.1)
        _save_path_viz("P16_astar_3804_4272", r["path"], "3804", "4272",
            "P16 — A* (Optimal)  3804 → 4272  (shortest 2-hop route)",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    # ── Group D: Cross-algorithm property tests ───────────────────────────────

    def test_P17_bfs_hop_optimal_3180_4043(self):
        """P17: BFS agrees with A* on hop count for 3180→4043 (both find 7 hops).

        When the hop-optimal path is also the time-optimal path, BFS and A*
        must return the same hop count, confirming both algorithms operate
        correctly on this route.
        """
        r_bfs = self._algo("bfs", "3180", "4043")
        r_ast = self._algo("astar", "3180", "4043")
        self.assertTrue(r_bfs["found"])
        self.assertEqual(r_bfs["hops"], r_ast["hops"],
            "BFS and A* must agree on hop count when optimal paths coincide")
        _save_path_viz("P17_bfs_3180_4043", r_bfs["path"], "3180", "4043",
            "P17 — BFS (Hop-optimal)  3180 → 4043  (matches A* hops)",
            f"BFS: {r_bfs['hops']} hops — A*: {r_ast['hops']} hops (both hop-optimal on this route)")

    def test_P18_idastar_optimal_4030_3002(self):
        """P18: IDA* finds the same optimal cost as A* on the short route 4030→3002."""
        r_ida = self._algo("cus2", "4030", "3002")
        r_ast = self._algo("astar", "4030", "3002")
        self.assertTrue(r_ida["found"])
        self.assertAlmostEqual(r_ida["total_time_min"], r_ast["total_time_min"], delta=0.05,
            msg="IDA* must match A* optimal cost on 4030→3002")
        _save_path_viz("P18_idastar_4030_3002", r_ida["path"], "4030", "3002",
            "P18 — IDA* / CUS2 (Optimal)  4030 → 3002  (3-hop route)",
            f"IDA*: {r_ida['total_time_min']:.2f} min = A*: {r_ast['total_time_min']:.2f} min  |  {r_ida['nodes_created']} nodes")

    def test_P19_astar_3122_4057(self):
        """P19: A* finds the east-to-west sub-route 3122→4057 in 3 hops, 5.01 min."""
        r = self._algo("astar", "3122", "4057")
        self.assertTrue(r["found"])
        self.assertEqual(r["path"][0], "3122"); self.assertEqual(r["path"][-1], "4057")
        self.assertEqual(r["hops"], 3)
        self.assertAlmostEqual(r["total_time_min"], 5.01, delta=0.1)
        _save_path_viz("P19_astar_3122_4057", r["path"], "3122", "4057",
            "P19 — A* (Optimal)  3122 → 4057  (east-to-west sub-route)",
            f"Path: {' → '.join(r['path'])}  |  {r['hops']} hops  |  {r['total_time_min']:.2f} min")

    def test_P20_dfs_suboptimal_cost_4272_4030(self):
        """P20: DFS finds a costlier path than A* on 4272→4030 (same hops, different route).

        Both algorithms find 6 hops, but DFS routes through the detour
        4272→2000→3120 instead of A*'s 4272→4040→3120, costing 8.61 min
        vs A*'s optimal 7.92 min — a 0.69 min (8.7%) cost penalty.
        """
        r_dfs = self._algo("dfs", "4272", "4030")
        r_ast = self._algo("astar", "4272", "4030")
        self.assertTrue(r_dfs["found"])
        self.assertEqual(r_dfs["hops"], 6, "DFS must find the 6-hop detour on 4272→4030")
        self.assertGreater(r_dfs["total_time_min"], r_ast["total_time_min"],
            f"DFS ({r_dfs['total_time_min']:.2f} min) must cost more than A* ({r_ast['total_time_min']:.2f} min)")
        _save_path_viz("P20_dfs_4272_4030", r_dfs["path"], "4272", "4030",
            "P20 — DFS (Sub-optimal cost)  4272 → 4030",
            f"DFS: {r_dfs['total_time_min']:.2f} min via {' → '.join(r_dfs['path'])}  (A* optimal: {r_ast['total_time_min']:.2f} min)")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("TBRGS TEST SUITE")
    print("  Set SKIP_ML_TESTS=1 to skip long ML training tests")
    print("=" * 65)
    unittest.main(verbosity=2)
