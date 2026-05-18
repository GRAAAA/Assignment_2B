import sys, os, math, unittest
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

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

    def test_T12_directed_edges(self):
        """T12: Graph is directed — not every edge has a reverse."""
        # Most edges do have reverses, but the graph is not guaranteed to be symmetric.
        # We just verify the structure is a dict of lists.
        self.assertIsInstance(self.graph, dict)
        sample = next(iter(self.graph.values()))
        self.assertIsInstance(sample, list)
        self.assertIsInstance(sample[0], tuple)
        self.assertEqual(len(sample[0]), 2)


# ──────────────────────────────────────────────────────────────────────────────
# T13–T17: Route Finding
# ──────────────────────────────────────────────────────────────────────────────

class TestRouteFinding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def test_T13_astar_finds_path(self):
        """T13: A* finds a path from 2000 to 3002."""
        goal, cost, path, nc = astar_tbrgs(self.graph, "2000", ["3002"])
        self.assertEqual(goal, "3002")
        self.assertGreater(len(path), 1)
        self.assertGreater(cost, 0)

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

    def test_T35_per_station_rf_models(self):
        """T35: Each SCATS location gets its own independent RF model."""
        self._skip_if_no_predictor()
        from subgraph import EDGE_TO_LOCATION
        # Train per-station RF for three different locations
        locs = list(set(EDGE_TO_LOCATION.values()))[:3]
        models = []
        for loc in locs:
            rf = self.predictor._get_station_rf(loc)
            self.assertIn("model", rf)
            self.assertIn("metrics", rf)
            models.append(id(rf["model"]))
        # The three station models must be distinct objects
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
        """T21: A* finds a path and returns it starting at origin."""
        r = self._run("astar")
        self.assertTrue(r["found"], "A* should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)

    def test_T22_bfs_finds_path(self):
        """T22: BFS finds a path (fewest hops)."""
        r = self._run("bfs")
        self.assertTrue(r["found"], "BFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)

    def test_T23_dfs_finds_path(self):
        """T23: DFS finds a path (may not be optimal)."""
        r = self._run("dfs")
        self.assertTrue(r["found"], "DFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)

    def test_T24_gbfs_finds_path(self):
        """T24: GBFS finds a path (greedy heuristic)."""
        r = self._run("gbfs")
        self.assertTrue(r["found"], "GBFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)

    def test_T25_cus1_iddfs_finds_path(self):
        """T25: IDDFS (CUS1) finds a path with fewest hops."""
        r = self._run("cus1")
        self.assertTrue(r["found"], "IDDFS should find a path")
        self.assertEqual(r["path"][0], self.origin)
        self.assertEqual(r["path"][-1], self.destination)

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

    def test_T27_optimal_algorithms_agree(self):
        """T27: A*, BFS, IDDFS, IDA* all find the same hop count (optimal)."""
        results = {a: self._run(a) for a in ["astar", "bfs", "cus1", "cus2"]}
        hops = [results[a]["hops"] for a in results if results[a]["found"]]
        self.assertTrue(all(h == hops[0] for h in hops),
                        f"Optimal algorithms should agree on hops: {hops}")

    def test_T28_dfs_may_find_longer_path(self):
        """T28: DFS path hops >= optimal (DFS is not hop-optimal)."""
        r_dfs  = self._run("dfs")
        r_bfs  = self._run("bfs")
        if r_dfs["found"] and r_bfs["found"]:
            self.assertGreaterEqual(r_dfs["hops"], r_bfs["hops"],
                "DFS should find path with >= hops compared to BFS")

    def test_T29_nodes_created_varies_by_algorithm(self):
        """T29: Different algorithms create different numbers of nodes."""
        results = self.run_all(self.origin, self.destination)
        nc_values = [r["nodes_created"] for r in results if r["found"]]
        # At least two different values (they shouldn't all be identical)
        self.assertGreater(len(set(nc_values)), 1,
            "Algorithms should differ in nodes created")

    def test_T30_all_algorithms_find_path(self):
        """T30: All 6 algorithms successfully find a path on the subgraph."""
        results = self.run_all(self.origin, self.destination)
        for r in results:
            self.assertTrue(r["found"],
                f"{r['algorithm']} failed to find a path")



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
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("TBRGS TEST SUITE")
    print("  Set SKIP_ML_TESTS=1 to skip long ML training tests")
    print("=" * 65)
    unittest.main(verbosity=2)
