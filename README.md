# TBRGS — Traffic-Based Route Guidance System
## COS30019 Introduction to AI — Assignment 2B

---

## Requirements

- Python 3.10 or above
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the data file in the **same folder** as all `.py` files:

```
Scats_Data_October_2006.xls
```

---

## Project Structure

```
├── Scats_Data_October_2006.xls   ← dataset (place here)
│
├── data_processor.py             ← data loading, reshaping, feature engineering
│
├── model_lstm.py                 ← LSTM model (standalone)
├── model_gru.py                  ← GRU model  (standalone)
├── model_rf.py                   ← Random Forest model (standalone)
├── ml_models.py                  ← aggregator: imports all 3, compare_models()
├── traffic_predictor.py          ← predict_flow(date, loc_index, time_slot, model)
│
├── subgraph.py                   ← 15-node SCATS subgraph + A* + visualisation
├── search_algorithms.py          ← adapter: runs all 6 A2A algorithms on subgraph
├── tbrgs_graph.py                ← full Boroondara graph + top-k paths
│
├── astar.py                      ← A* Search          (from Assignment 2A)
├── bfs.py                        ← Breadth-First Search
├── dfs.py                        ← Depth-First Search
├── gbfs.py                       ← Greedy Best-First Search
├── cus1.py                       ← IDDFS (CUS1)
├── cus2.py                       ← IDA*  (CUS2)
│
├── main_gui.py                   ← tkinter GUI — entry point
├── test_tbrgs.py                 ← 39 test cases (T01–T35)
│
├── requirements.txt
└── README.md
```

---

## Running the GUI

```bash
python main_gui.py
```

The GUI is a **single-screen interface** with three areas:

**INPUT (left panel)**
- Start Node / End Node — choose from the 15 subgraph SCATS sites
- Day / Time — day of week and 15-minute time slot
- Prediction Model — which ML model predicts traffic (best / lstm / gru / rf)
- Search Algorithm — which path-finding algorithm to use

**GRAPH VISUALISATION (right panel)**
- Displays the 15-node Boroondara subgraph
- The optimal path is highlighted in amber after a route is found

**RESULTS (bottom panel)**
- Best ML model and its metrics (RMSE / NRMSE / MAE / R2)
- Predicted traffic flow / weight on each edge of the route
- The optimal path and its total weight (travel time)
- A comparison of all six search algorithms

### How to use

1. Click **Train ML Models**. Training is fully automatic — the models
   (LSTM, GRU, Random Forest) learn the general daily traffic pattern
   from the Boroondara SCATS data. Training runs in the background and
   takes a few minutes.
2. Once trained, choose **Start / End / Day / Time / Model / Algorithm**
   and click **Find Optimal Route**.
3. The optimal path appears on the graph, and the full results
   (predicted flow per edge, best path, total weight, algorithm
   comparison, best ML model) appear in the Results panel.

Note: each graph edge is weighted by the predicted traffic flow at its
starting SCATS site, in the direction of travel. The predicted traffic
flow is converted into estimated travel time, which is used as the edge
weight for route optimisation.

## Running Subgraph Standalone

```bash
# Default: 4030 → 4043
python subgraph.py

# Custom origin → destination
python subgraph.py 4030 4273
```

## Running All 6 Algorithms Standalone

```bash
# Default: 4030 → 4043
python search_algorithms.py

# Custom
python search_algorithms.py 4030 4273
```

### A note on the six algorithms

All six Assignment-2A algorithms run on the weighted subgraph, but they
serve different purposes:

- **A\*** and **IDA\* (CUS2)** are *weight-aware*: they minimise total
  travel time, so the route they return is the genuinely fastest path.
  A\* is the system default.
- **GBFS** uses the straight-line heuristic only — fast, but not guaranteed
  optimal on travel time.
- **BFS**, **DFS**, and **IDDFS (CUS1)** are *uninformed* and **ignore edge
  weights**. They find *a* connecting path (BFS by fewest hops), not the
  fastest one. They are included for algorithm comparison, as required by
  Assignment 2A — not because they optimise traffic.

In short: **A\* / IDA\* for optimal traffic routing; BFS / DFS for
comparison.**

---

## Running Tests

```bash
# Fast (no ML training, ~2 seconds):
SKIP_ML_TESTS=1 python test_tbrgs.py

# Full suite (includes ML training, ~2–5 minutes):
python test_tbrgs.py
```

39 test cases covering:
- T01–T05  : Data processing
- T06–T09  : Flow → speed → travel time conversion
- T10–T12  : Graph building and edge weights
- T13–T17  : Route finding (A*, top-k, edge cases)
- T18–T20  : ML model prediction sanity
- T21–T30  : All 6 A2A algorithms on the subgraph
- T31–T35  : Integration checks (per-edge flow, per-station RF)

---

## ML Models

The system trains three models and selects the best one **automatically**
from the actual training run — there is no pre-set winner. Selection is
data-driven: `compare_models()` picks the model with the lowest NRMSE on the
test split, so the "best" model can differ between runs depending on the
training location, epochs, and the train/test split.

| Model | Type | Typical RMSE range |
|-------|------|--------------------|
| LSTM | Deep learning (sequence) | ~15–25 |
| GRU | Deep learning (sequence) | ~15–25 |
| Random Forest | Ensemble (per-station) | ~15–25 |

*Ranges are indicative only. The three models perform comparably on this
dataset; the actual metrics and the selected best model are reported in the
Results panel after each training run. Do not assume a fixed winner.*

### Per-station Random Forest models

Random Forest trains in seconds, so the system trains an **independent RF
model for every SCATS location** used by the subgraph edges (30 directed
edges → 30 per-station models). Each model learns the exact traffic profile
of its own road and direction.

LSTM and GRU are **not** trained per-station: deep models take minutes each,
and they learn the *general* daily pattern (morning/evening peaks) that is
consistent across intersections. They are trained once on a representative
arterial location and applied to each road's own historical sequence at
prediction time. This gives per-station rigour for RF while keeping total
training time practical.

---

## Flow → Travel Time Formula

```
flow_per_hr = predicted_flow × 4

A = −1.4648375    B = 93.75
discriminant = B² + 4·A·flow_per_hr

Under capacity (flow ≤ 1500 veh/hr):
    speed = (−B − √disc) / 2A   [capped at 60 km/h]

Over capacity:
    speed = (−B + √disc) / 2A

edge_weight = (distance_km / speed_km_h) × 60 + 0.5   [minutes]
```

Assumptions (from assignment spec):
- Speed limit: 60 km/h on all links
- Flow at capacity: 1500 veh/hr, speed at capacity: 32 km/h
- Intersection delay: 30 seconds (0.5 min) per node

---

## Subgraph Nodes (15 SCATS Sites)

| SCATS ID | Location |
|----------|----------|
| 3120 | BURKE_RD / CANTERBURY_RD |
| 3122 | CANTERBURY_RD / STANHOPE_GV |
| 3127 | BALWYN_RD / CANTERBURY_RD |
| 3180 | DONCASTER_RD / BALWYN_RD |
| 3804 | TRAFALGAR_RD / RIVERSDALE_RD |
| 4030 | BURKE_RD / DONCASTER_RD |
| 4032 | BURKE_RD / HARP_RD |
| 4034 | BURKE_RD / WHITEHORSE_RD |
| 4035 | BURKE_RD / MONT_ALBERT_RD |
| 4040 | BURKE_RD / RIVERSDALE_RD |
| 4043 | BURKE_RD / TOORAK_RD |
| 4057 | BALWYN_RD / BELMORE_RD |
| 4063 | BALWYN_RD / WHITEHORSE_RD |
| 4272 | RIVERSDALE_RD / TOORONGA_RD |
| 4273 | TOORONGA_RD / TOORAK_RD |

The subgraph forms three intersecting corridors:

- **Burke Road** (N→S): 4030 → 4032 → 4034 → 4035 → 3120 → 4040 → 4043
- **Canterbury Road** (W→E, branches at 3120): 3120 → 3122 → 3127
- **Balwyn Road** (S→N, branches at 3127): 3127 → 4063 → 4057 → 3180
- **Riversdale / Tooronga** (from 4040): 4040 → 3804, and
  4040 → 4272 → 4273 → 4043
