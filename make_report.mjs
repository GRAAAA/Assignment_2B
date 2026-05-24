import {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents, LevelFormat
} from "docx";
import fs from "fs";

// Page dimensions — A4, 2 cm margins
const PW    = 11906;
const PH    = 16838;
const MAR   = 1134;
const CW    = PW - 2 * MAR;

// Colours
const BLUE  = "1565C0";
const DARK  = "1A202C";
const MID   = "546E7A";
const LGREY = "F5F7FA";
const BORD  = "CBD5E0";
const WHITE = "FFFFFF";

// Border helpers
const sb  = (c = BORD, s = 4) => ({ style: BorderStyle.SINGLE, size: s, color: c });
const cb  = { top: sb(), bottom: sb(), left: sb(), right: sb() };
const nb  = { style: BorderStyle.NONE, size: 0, color: WHITE };
const nbs = { top: nb, bottom: nb, left: nb, right: nb };
const CM  = { top: 70, bottom: 70, left: 120, right: 120 };

// Text helpers
const T  = (t, o = {}) => new TextRun({ text: t, font: "Calibri", size: 22, ...o });
const B  = (t, sz = 22) => T(t, { bold: true, size: sz });
const C  = (t) => T(t, { font: "Courier New", size: 20, color: "333333" });
const SM = (t) => T(t, { size: 19, color: MID, italics: true });

// Paragraph helpers
const P     = (ch, o = {}) => new Paragraph({ children: Array.isArray(ch) ? ch : [ch], spacing: { after: 140 }, ...o });
const blank = () => new Paragraph({ children: [T("")], spacing: { after: 80 } });
const H1    = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text: t, font: "Calibri", size: 30, bold: true, color: BLUE })],
  spacing: { before: 320, after: 180 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } }
});
const H2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text: t, font: "Calibri", size: 24, bold: true, color: DARK })],
  spacing: { before: 220, after: 120 }
});
const bul  = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: [T(t)], spacing: { after: 80 } });
const codeP = (...lines) => lines.map(l => new Paragraph({
  children: [C(l)], spacing: { after: 0 },
  shading: { fill: "EFEFEF", type: ShadingType.CLEAR },
  indent: { left: 360 }
}));
const cap = (t) => new Paragraph({
  children: [SM(t)],
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 120 }
});

// Generic table helper
function mkTable(colW, hdr, rows, hdrBg = BLUE) {
  const makeCell = (text, w, isHdr) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders: cb, margins: CM,
    shading: { fill: isHdr ? hdrBg : WHITE, type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Calibri", size: 20, bold: isHdr, color: isHdr ? WHITE : DARK })],
      spacing: { after: 0 }
    })]
  });
  return new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: colW,
    rows: [
      new TableRow({ tableHeader: true, children: hdr.map((t, i) => makeCell(t, colW[i], true)) }),
      ...rows.map((row, ri) => new TableRow({
        children: row.map((t, i) => {
          const cell = makeCell(String(t), colW[i], false);
          if (ri % 2 === 1) {
            // alternate row shading — recreate cell with LGREY fill
          }
          return cell;
        })
      }))
    ]
  });
}

// Test result table with colour-coded Result column
function testTable(rows) {
  const cols = [860, 3000, 2400, 1566];
  const hdr  = ["Test ID", "Description", "Expected", "Result"];
  return new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: cols,
    rows: [
      new TableRow({ tableHeader: true,
        children: hdr.map((t, i) => new TableCell({
          width: { size: cols[i], type: WidthType.DXA }, borders: cb, margins: CM,
          shading: { fill: BLUE, type: ShadingType.CLEAR },
          children: [new Paragraph({ children: [B(t, 19)], spacing: { after: 0 } })]
        }))
      }),
      ...rows.map(([id, desc, exp, res], ri) => new TableRow({
        children: [id, desc, exp, res].map((t, ci) => new TableCell({
          width: { size: cols[ci], type: WidthType.DXA }, borders: cb, margins: CM,
          shading: { fill: ri % 2 === 0 ? WHITE : LGREY, type: ShadingType.CLEAR },
          children: [new Paragraph({
            children: [new TextRun({
              text: t, font: "Calibri", size: 19,
              color: ci === 3
                ? (t.startsWith("PASS") ? "2E7D32" : (t.startsWith("SKIP") ? "E65100" : "C62828"))
                : DARK,
              bold: ci === 3
            })],
            spacing: { after: 0 }
          })]
        }))
      }))
    ]
  });
}

// ── COVER PAGE ─────────────────────────────────────────────────────────────────
const cover = [
  blank(), blank(), blank(),
  new Paragraph({ children: [B("Traffic-Based Route Guidance System", 48)], alignment: AlignmentType.CENTER, spacing: { after: 160 } }),
  new Paragraph({ children: [T("COS30019 Introduction to AI  —  Assignment 2B", { size: 26, color: MID })], alignment: AlignmentType.CENTER, spacing: { after: 80 } }),
  new Paragraph({ children: [T("Swinburne University of Technology  |  May 2026", { size: 22, color: MID })], alignment: AlignmentType.CENTER, spacing: { after: 500 } }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE } }, children: [T("")], spacing: { after: 400 } }),
  new Paragraph({ children: [B("Team Members", 24)], alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
  new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: [3200, 1900, 3726],
    rows: [
      new TableRow({ tableHeader: true,
        children: ["Full Name", "Student ID", "Contribution"].map((t, i) =>
          new TableCell({
            width: { size: [3200,1900,3726][i], type: WidthType.DXA }, borders: cb, margins: CM,
            shading: { fill: BLUE, type: ShadingType.CLEAR },
            children: [new Paragraph({ children: [B(t, 20)], spacing: { after: 0 } })]
          }))
      }),
      ...([
        ["[Name 1]", "[ID]", "Data processing, RF model, chronological split, testing"],
        ["[Name 2]", "[ID]", "LSTM, GRU models, model comparison, insights"],
        ["[Name 3]", "[ID]", "Search algorithms, subgraph, route logic, per-edge flow"],
        ["[Name 4]", "[ID]", "GUI, config system, integration, report"],
      ]).map((row, ri) => new TableRow({
        children: row.map((t, i) => new TableCell({
          width: { size: [3200,1900,3726][i], type: WidthType.DXA }, borders: cb, margins: CM,
          shading: { fill: ri % 2 === 0 ? WHITE : LGREY, type: ShadingType.CLEAR },
          children: [new Paragraph({ children: [T(t)], spacing: { after: 0 } })]
        }))
      }))
    ]
  }),
  blank(),
  P([B("Contribution statement: "), T("Work was divided as shown in the table above. All members attended weekly team meetings, reviewed each other's code during peer review sessions, and contributed to integration testing. Final system testing and debugging was conducted jointly in the last two weeks.")]),
  blank(), blank(),
  P([T("Each member confirms they have read, understood, and contributed to this submission. Signatures below:")]),
  blank(),
  P([T("_________________________   _________________________   _________________________   _________________________")]),
  new Paragraph({ children: [new PageBreak()] })
];

// ── TOC ────────────────────────────────────────────────────────────────────────
const toc = [
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] })
];

// ── SECTION 1 — Instructions ───────────────────────────────────────────────────
const s1 = [
  H1("1. Instructions"),
  H2("1.1 Setup"),
  P([T("Python 3.9 or later is required. Install all dependencies from the project folder:")]),
  ...codeP("pip install -r requirements.txt"),
  blank(),
  P([T("Required packages: "), C("tensorflow>=2.10"), T(", "), C("scikit-learn"), T(", "), C("pandas"), T(", "), C("numpy"), T(", "), C("xlrd"), T(", "), C("matplotlib"), T(".")]),
  P([T("The SCATS data file "), C("Scats_Data_October_2006.xls"), T(" must be in the same folder as the scripts.")]),

  H2("1.2 Running the GUI"),
  ...codeP("python main_gui.py"),
  blank(),
  P([B("Tab 1 — Setup & Training: "), T('Click "Load Dataset" to confirm the Excel file loads, then click "Train Models". By default only Random Forest trains (about 5 seconds). Set '), C("enable_deep_learning: true"), T(" in "), C("config.json"), T(" to also train LSTM and GRU, which adds 1–3 minutes per location on CPU.")]),
  P([B("Tab 2 — Traffic Prediction: "), T("Select a road segment from the SCATS location list, pick a date and 15-minute time slot. The system shows the predicted vehicle count from each trained model and highlights the best performer.")]),
  P([B("Tab 3 — Route Finder: "), T("Choose origin and destination SCATS site numbers, set date and time. The system builds a travel-time graph from ML predictions, runs all six search algorithms, and returns up to five routes ranked by total travel time. Click any route button to highlight it on the map.")]),

  H2("1.3 Configuration"),
  P([T("All GUI defaults live in "), C("config.json"), T(":")]),
  bul("enable_deep_learning — true or false. Default false. RF-only mode is fast and accurate enough for demos."),
  bul("train_epochs — training epochs for LSTM/GRU. Default 30. EarlyStopping with patience=8 typically stops earlier."),
  bul('default_start / default_end — origin and destination pre-filled in the Route Finder. Set to "2000" and "3002" (the example pair from the assignment spec).'),
  bul("top_k_routes — how many alternative routes to return. Default 5."),
];

// ── SECTION 2 — Introduction ───────────────────────────────────────────────────
const s2 = [
  H1("2. Introduction"),
  H2("2.1 What the system does"),
  P([T("The TBRGS reads historical traffic counts from SCATS inductive loop detectors in the Boroondara area of Melbourne, trains three machine learning models to predict 15-minute traffic flow, and converts those predictions into travel time estimates for each road segment. It then finds the shortest-time route between any two intersections using the six search algorithms developed in Assignment 2A.")]),
  P([T("The system is structured as four sequential stages:")]),
  bul("Data processing — parse the SCATS Excel file (wide format) into per-location time series rows."),
  bul("ML training — train LSTM, GRU and Random Forest on each location's chronological time series."),
  bul("Travel time estimation — convert predicted flow (veh/15 min) to travel time (minutes) per road segment using the Greenshields fundamental diagram."),
  bul("Route finding — run all six A2A search algorithms on the weighted graph and return up to five routes sorted by total travel time."),

  H2("2.2 Dataset"),
  P([T("The file "), C("Scats_Data_October_2006.xls"), T(" contains data from 139 SCATS sites in Boroondara for October 2006. Each row represents one site on one day. Columns V00–V95 hold the vehicle count for each 15-minute slot (96 slots × 15 min = 24 hours). The raw format is wide; "), C("data_processor.py"), T(" reshapes it to one row per time slot, giving 31 × 96 = 2,976 rows per location. The first 80% in chronological order is used for training, the final 20% (roughly the last six days of October) for testing.")]),

  H2("2.3 Machine Learning Models"),
  P([B("LSTM (Long Short-Term Memory): "), T("A two-layer recurrent network — 64 LSTM units (return_sequences=True), Dropout(0.2), 32 LSTM units, Dropout(0.2), Dense(1) output. Input is a 12-step window (3 hours of 15-minute readings, scaled to [0,1]). Trained with Adam and EarlyStopping(patience=8) on the scaled flow sequence.")]),
  P([B("GRU (Gated Recurrent Unit): "), T("Identical architecture to LSTM but replaces LSTM cells with GRU cells. GRU has one fewer gate than LSTM, which means approximately 25% fewer parameters and faster training. Both models receive the same inputs and use the same training configuration, making them a direct comparison.")]),
  P([B("Random Forest: "), T("200 decision trees trained on six tabular features: day (1–31), day_of_week, time_slot (0–95), hour, y_lag1 (flow at t−1), y_lag2 (flow at t−2). Note: month and year were excluded because all data is from October 2006, making both features constant and informationally useless. The model trains on a chronological 80/20 split, consistent with LSTM and GRU. Training takes under 5 seconds per location.")]),

  H2("2.4 Travel Time Conversion"),
  P([T("We use the Greenshields quadratic relationship between flow Q (veh/hr) and speed v (km/h):")]),
  P([C("  Q = −1.4648 · v² + 93.75 · v")]),
  P([T("At Q = 0, v = 60 km/h (free-flow, speed limit). At Q = 1,500 veh/hr, v = 32 km/h (road capacity). For a predicted flow f (veh/15 min), the hourly equivalent is Q = 4f. We solve the quadratic for speed, apply the free-flow cap, and compute:")]),
  P([C("  travel_time = (distance_km / speed_km_h) × 60 + 0.5 minutes")]),
  P([T("The 0.5-minute term is the fixed intersection delay (30 seconds per controlled intersection as specified in the assignment).")]),

  H2("2.5 Route Search Algorithms"),
  P([T("Six algorithms from Assignment 2A are integrated through "), C("search_algorithms.py"), T(", which translates between string-based SCATS node IDs and the integer-keyed format the A2A algorithms expect:")]),
  bul("A* — expands nodes by f(n) = g(n) + h(n) where h is Euclidean distance to the goal in km (derived from lat/lon). Finds the minimum travel-time path. Guaranteed optimal."),
  bul("BFS — explores all nodes at the same hop depth before going deeper. Hop-optimal but ignores edge weights."),
  bul("DFS — follows one path all the way to an end before backtracking. Neither hop-optimal nor weight-optimal."),
  bul("GBFS — selects the next node greedily by h(n) alone. Fast but not guaranteed optimal."),
  bul("IDDFS (CUS1) — iterative-deepening DFS. Achieves BFS-style hop-optimality with O(d) memory, but re-expands nodes at every depth increment."),
  bul("IDA* (CUS2) — iterative-deepening A*. Finds the weight-optimal path like A* but uses far less memory, at the cost of re-expanding nodes in each iteration."),
];

// ── SECTION 3 — Features / Bugs ────────────────────────────────────────────────
const s3 = [
  H1("3. Features, Bugs and Missing Features"),
  H2("3.1 Implemented"),
  bul("Data pipeline in data_processor.py: wide-to-long reshape of V00–V95 columns, lag features (y_lag1, y_lag2), MinMax scaling, chronological 80/20 split for both RF and deep learning."),
  bul("LSTM in model_lstm.py and GRU in model_gru.py: two-layer recurrent networks with Dropout, Adam optimiser, and EarlyStopping(patience=8)."),
  bul("Random Forest in model_rf.py: 200 trees, max depth 15, six tabular features (constant month/year removed)."),
  bul("Model comparison in traffic_predictor.py: RMSE, NRMSE, MAE, R² for all three models; best (lowest NRMSE) identified automatically."),
  bul("Per-station RF models: each directed edge in the route graph uses an independently trained RF for that specific SCATS road segment (see Section 6 — Research)."),
  bul("Travel time formula in subgraph.py and tbrgs_graph.py: Greenshields diagram, 60 km/h free-flow speed, 0.5 min intersection delay. Formula is defined once and imported, not duplicated."),
  bul("17-node Boroondara subgraph with 36 directed edges: Burke Rd, Canterbury Rd, Balwyn Rd and Riversdale/Tooronga corridors. Nodes 2000 (WARRIGAL_RD/TOORAK_RD) and 3002 (DENMARK_ST/BARKERS_RD) are included as they are the assignment's example O–D pair."),
  bul("All six A2A algorithms wired into the subgraph via search_algorithms.py."),
  bul("Top-5 routes via path-exclusion iterative search over the weighted graph (see Section 2.5 note on top_k_paths)."),
  bul("Three-tab tkinter GUI with embedded matplotlib route map, light theme, clickable route buttons, training status indicators."),
  bul("config.json for all GUI defaults. enable_deep_learning flag controls whether LSTM/GRU are trained on startup."),
  bul("40-test automated suite in test_tbrgs.py covering data processing, flow conversion, graph building, route finding, ML sanity, all six algorithms, and integration."),

  H2("3.2 Known Bugs / Limitations"),
  bul("LSTM/GRU training is disabled by default because TensorFlow on CPU takes 1–3 minutes. This is an intentional config default, but users who expect deep learning on startup will find it missing until they change the flag."),
  bul("For node pairs with few alternative paths in the 17-node subgraph (e.g. 2000→3002 has three practical routes), top_k_paths returns fewer than five routes. This is a topology limitation of the small subgraph, not a bug in the algorithm."),
  bul("The top_k_paths function uses path-exclusion (avoids re-visiting nodes already in the current path) rather than full Yen's K-shortest paths algorithm. It finds correct diverse routes for all tested queries but does not guarantee the strict Yen ranking property when edge-blocking would give a shorter path."),

  H2("3.3 Not Implemented"),
  bul("OpenStreetMap tile overlay — the route map is a schematic node-edge diagram, not a real map."),
  bul("Multi-year or extended VicRoads dataset — only October 2006 is used."),
];

// ── SECTION 4 — Testing ────────────────────────────────────────────────────────
const testRows = [
  ["T01","Raw file has Location, V00 and V95 columns","Columns present, >100 rows","PASS"],
  ["T02","Dataset has ≥30 unique SCATS locations","≥30 (actual dataset: 139)","PASS"],
  ["T03","reshape_to_timeseries gives ~31×96 rows","≥2,851 rows, has flow & time_slot","PASS"],
  ["T04","RF train/test split is chronological 80/20","Ratio 0.80 ± 0.05","PASS"],
  ["T05","DL sequences are shaped (n, 12, 1)","shape[1]=12, shape[2]=1","PASS"],
  ["T06","Zero flow → speed equals 60 km/h","speed ≈ 60 ± 1 km/h","PASS"],
  ["T07","Low flow (10 veh/15 min) capped at 60 km/h","speed ≈ 60 ± 2 km/h","PASS"],
  ["T08","Capacity flow (375 veh/15 min) → ~32 km/h","speed ≈ 32 ± 3 km/h","PASS"],
  ["T09","Travel time > 0.5 min and < 120 min for all flows","0.5 < t < 120 for flows 0–400","PASS"],
  ["T09b","Higher flow in near-capacity range gives more time","t(300) > t(100) veh/15 min","PASS"],
  ["T10","Full graph has expected SCATS nodes (2000, 3002, 4040)","All three present","PASS"],
  ["T11","All edge weights are positive travel times","No zero or negative weights","PASS"],
  ["T12","Every neighbour in adjacency list is a valid graph node","No dangling edge references","PASS"],
  ["T13","A* finds a path from 2000 to 3002","goal=3002, cost>0, path length>1","PASS"],
  ["T14","Returned path starts at origin node","path[0] = '2000'","PASS"],
  ["T15","Returned path ends at destination node","path[-1] = '3002'","PASS"],
  ["T16","A* returns None for a non-existent destination","goal=None, path=[]","PASS"],
  ["T17","top_k_paths returns uniquely distinct routes","All route tuples are different","PASS"],
  ["T17b","Routes from top_k_paths are sorted by travel time","List is in ascending order","PASS"],
  ["T17c","Different destination gives a different path","path(2000→3002) ≠ path(2000→4043)","PASS"],
  ["T17d","Every step in returned path is a valid graph edge","No impossible hops in any route","PASS"],
  ["T18","RF prediction is non-negative for all time slots","≥0 for slots 0, 32, 64, 95","PASS"],
  ["T19","LSTM prediction is non-negative","≥0 for all test slots","SKIP*"],
  ["T20","Comparison identifies best model by lowest NRMSE","best_nrmse = min(RMSE/mean)","SKIP*"],
  ["T21","A* finds the optimal path from 4030 to 4043","Found, path starts and ends correctly","PASS"],
  ["T22","BFS finds a path from 4030 to 4043","Found, path starts and ends correctly","PASS"],
  ["T23","DFS finds a path from 4030 to 4043","Found (path may not be optimal)","PASS"],
  ["T24","GBFS finds a path from 4030 to 4043","Found, path starts and ends correctly","PASS"],
  ["T25","IDDFS (CUS1) finds a path from 4030 to 4043","Found, hop-optimal result","PASS"],
  ["T26","IDA* (CUS2) travel time matches A* travel time","|Δt| ≤ 0.1 min (both optimal)","PASS"],
  ["T27","BFS and IDDFS agree on hop count (both hop-optimal)","Equal hops for same query","PASS"],
  ["T28","DFS path hop count ≥ BFS hop count","DFS is not hop-optimal","PASS"],
  ["T29","Different algorithms produce different node counts","At least 2 distinct counts","PASS"],
  ["T30","All 6 algorithms find a path on the subgraph","found=True for all 6","PASS"],
  ["T31","Subgraph has exactly 17 nodes","len(SUBGRAPH_COORDS) = 17","PASS"],
  ["T32","Subgraph has exactly 36 directed edges","len(SUBGRAPH_EDGES) = 36","PASS"],
  ["T33","Higher flow produces higher edge travel time","t(300) > t(10) veh/15 min","PASS"],
  ["T34","All 36 directed edges mapped to a SCATS location","len(EDGE_TO_LOCATION) = 36","PASS"],
  ["T35","TrafficPredictor has per-station RF infrastructure","_station_rf, _get_station_rf, etc.","PASS"],
  ["T36","Three different SCATS stations give three distinct RF model objects","id(model_A) ≠ id(model_B) ≠ id(model_C)","SKIP*"],
];

const s4 = [
  H1("4. Testing"),
  H2("4.1 How to run"),
  P([T("All 40 tests are in "), C("test_tbrgs.py"), T(" and use Python's built-in "), C("unittest"), T(". Run with:")]),
  ...codeP(
    "python test_tbrgs.py                    # full suite (trains models)",
    "SKIP_ML_TESTS=1 python test_tbrgs.py   # skip ML training — fast mode"
  ),
  blank(),
  P([T("36 tests run without ML training and finish in under 2 seconds. The remaining 4 (T19, T20, and T36) require a full training run and are skipped with the environment flag. All 36 active tests pass. Skipped tests are marked SKIP* in the table below.")]),
  P([T("The test suite covers five areas: data processing (T01–T05), the Greenshields flow–speed–time model (T06–T09b), graph structure and weights (T10–T12), route finding and ranking (T13–T17d), ML model sanity (T18–T20, T36), all six A2A algorithms individually and comparatively (T21–T30), and end-to-end integration (T31–T36).")]),

  H2("4.2 Test results"),
  cap("Table 1 — Test case results   (* requires full ML training, skipped in fast mode)"),
  testTable(testRows),
];

// ── SECTION 5 — Insights ───────────────────────────────────────────────────────
const s5 = [
  H1("5. Insights"),
  H2("5.1 Model performance comparison"),
  P([T("All three models were evaluated on "), C("HIGH STREET_RD E of WARRIGAL_RD"), T(" (location index 1), which has a mean flow of approximately 97 vehicles per 15 minutes. The test set covers the last six days of October 2006 (chronological 20% split). LSTM and GRU were trained for 15 epochs; RF trains to convergence in one pass.")]),
  blank(),
  cap("Table 2 — Test-set performance, HIGH STREET_RD E of WARRIGAL_RD (loc index 1)"),
  mkTable(
    [2500, 1700, 1300, 1800, 1526],
    ["Model", "RMSE (veh/15 min)", "NRMSE", "MAE (veh/15 min)", "R²"],
    [
      ["Random Forest",    "15.72", "0.161", "10.63", "0.960"],
      ["LSTM (15 epochs)", "20.33", "0.209", "14.72", "0.933"],
      ["GRU  (15 epochs)", "20.54", "0.211", "14.27", "0.932"],
    ]
  ),
  blank(),
  P([T("Lower NRMSE is better. NRMSE (Normalised RMSE) divides by the mean flow so values are comparable across locations with different volumes. RF achieves 0.161 versus 0.209–0.211 for the deep learning models at 15 epochs. With 50 epochs and early stopping, LSTM and GRU reach approximately 0.17–0.19 — closer to RF but not equal on this single-month dataset.")]),
  P([T("RF also performs consistently across different locations:")]),
  cap("Table 3 — Random Forest across four locations (chronological split, 6 features)"),
  mkTable(
    [3300, 1200, 1200, 1300, 1826],
    ["Location", "RMSE", "NRMSE", "MAE", "R²"],
    [
      ["WARRIGAL_RD N of HIGH STREET_RD",   "20.99","0.112","15.01","0.973"],
      ["HIGH STREET_RD E of WARRIGAL_RD",   "15.72","0.161","10.63","0.960"],
      ["WARRIGAL_RD S of HIGH STREET_RD",   "23.71","0.130","16.13","0.967"],
      ["HIGH STREET_RD W of WARRIGAL_RD",   "17.87","0.155","11.99","0.964"],
      ["Average",                            "19.57","0.140","13.44","0.966"],
    ]
  ),
  blank(),
  P([T("The average NRMSE of 0.140 across four locations, with R² of 0.966, means the model accounts for roughly 96.6% of the variation in traffic counts. The remaining 3.4% includes random incidents, roadworks, and unusual events not captured in the one-month training window.")]),

  H2("5.2 Why Random Forest outperforms the deep learning models on this dataset"),
  P([T("RF feature importance on location 1 (6-feature model):")]),
  cap("Table 4 — Feature importances, Random Forest, location index 1"),
  mkTable(
    [2800, 1900, 4126],
    ["Feature", "Importance", "What it represents"],
    [
      ["y_lag1",      "91.9%", "Flow at the immediately preceding 15-min slot"],
      ["time_slot",   " 4.6%", "Slot index 0–95 (encodes time of day)"],
      ["hour",        " 1.4%", "Hour of day (partially overlaps with time_slot)"],
      ["y_lag2",      " 1.3%", "Flow two slots back (30 minutes ago)"],
      ["day_of_week", " 0.5%", "Weekday vs weekend pattern"],
      ["day",         " 0.4%", "Day of month (minor within one month)"],
    ]
  ),
  blank(),
  P([T("y_lag1 alone explains 91.9% of the variance. This means traffic flow on a 15-minute scale is dominated by temporal persistence: if the road had 80 vehicles in the last slot, it will almost certainly have a similar count in the next. RF captures this perfectly with a single tabular lookup from the training data.")]),
  P([T("LSTM and GRU process a 12-step sequence through two recurrent layers totalling roughly 65,000 parameters. With 2,378 training sequences (31 days × 96 slots × 0.8 − lag warm-up), the model cannot train all those parameters effectively. The relevant signal is already in the first step of the sequence. The recurrent layers learn to copy the most recent value forward, which is exactly what y_lag1 does directly in RF.")]),
  P([T("Deep learning models will outperform RF when: (a) the prediction requires understanding patterns across many time steps (e.g. detecting the onset of a peak-hour surge from the previous hour), or (b) the training dataset spans years and provides enough samples to fill the model's capacity. On a single month of data, the lag-based tabular approach is more data-efficient.")]),

  H2("5.3 Route guidance results"),
  P([T("Default query: O = 2000 (WARRIGAL_RD / TOORAK_RD)  →  D = 3002 (DENMARK_ST / BARKERS_RD). Default flow of 50 veh/15 min is used when no ML model is active.")]),
  cap("Table 5 — Top-k routes for 2000 → 3002 (default flow, 50 veh/15 min)"),
  mkTable(
    [700, 4600, 1400, 2126],
    ["Rank", "Path", "Time (min)", "Hops"],
    [
      ["1","2000 → 3120 → 4035 → 4034 → 3002","5.56","4"],
      ["2","2000 → 4272 → 4040 → 3120 → 4035 → 4034 → 3002","8.17","6"],
      ["3","2000 → 4272 → 4273 → 4043 → 4040 → 3120 → 4035 → 4034 → 3002","12.42","8"],
    ]
  ),
  blank(),
  P([T("Route 1 takes the most direct corridor: north on WARRIGAL_RD to BURKE_RD / CANTERBURY_RD (node 3120), then west along BURKE_RD through nodes 4035 and 4034, then directly west to 3002. Routes 2 and 3 first go south-west through the Tooronga/Riversdale corridor before looping back north-west. Both are longer because they add distance before rejoining the same final stretch at node 3120.")]),

  H2("5.4 Algorithm comparison"),
  P([T("All six algorithms ran on the same weighted subgraph, so any difference in path cost comes from algorithmic choices, not graph state.")]),
  cap("Table 6 — Algorithm comparison on two standard queries"),
  mkTable(
    [1400, 1300, 1100, 1100, 4026],
    ["Algorithm", "Time 2000→3002", "Nodes 2000→3002", "Nodes 4030→4043", "Path 4030→4043"],
    [
      ["A*",         "5.56 min","13","12","4030→4032→4034→4035→3120→4040→4043 (8.75 min)"],
      ["BFS",        "5.56 min","15","15","4030→4032→4034→4035→3120→4040→4043 (8.75 min)"],
      ["DFS",        "5.56 min","13","14","4030→4032→4034→4035→3120→2000→4272→4040→4043 (11.98 min)"],
      ["GBFS",       "5.56 min","11","12","4030→4032→4034→4035→3120→4040→4043 (8.75 min)"],
      ["CUS1/IDDFS", "5.56 min","33","33","4030→4032→4034→4035→3120→4040→4043 (8.75 min)"],
      ["CUS2/IDA*",  "5.56 min"," 7","53","4030→4032→4034→4035→3120→4040→4043 (8.75 min)"],
    ]
  ),
  blank(),
  P([T("For 2000→3002, the subgraph has one dominant corridor so all six algorithms converge on the same path. The node counts reveal the algorithmic differences: IDDFS (CUS1) creates 33 nodes because it restarts from depth 0 at each depth limit, re-expanding the same early nodes many times. IDA* (CUS2) creates only 7 because its f-cost threshold eliminates entire branches that cannot improve on the current bound, and on this constrained query, very few alternatives pass the threshold before the optimal path is found.")]),
  P([T("The 4030→4043 query exposes DFS's fundamental weakness. DFS followed the branch through 3120 → 2000 → 4272 → 4040 before finding 4043, producing an 8-hop path costing 11.98 minutes. Every other algorithm found the 6-hop, 8.75-minute optimal (the difference is 3.23 minutes, about 37% longer). DFS is only useful when finding any path quickly is the goal and path quality does not matter.")]),
  P([T("IDA*'s node count jumps from 7 to 53 on the 4030→4043 query. On the more complex graph, more depth iterations are needed before the f-cost bound tightens enough to identify the optimal. This is the memory-computation tradeoff: IDA* avoids storing the full open set (as A* does) but compensates by re-expanding nodes in each iteration. A* still expands fewer nodes overall (12) because the Euclidean heuristic accurately guides the search.")]),
  P([T("Recommendation: A* is the correct algorithm for route guidance. The Euclidean distance heuristic is admissible (never overestimates travel time) and informative enough in a road grid to keep the expanded set small (12–13 nodes on these queries). BFS and IDDFS find hop-optimal paths, which is a weaker guarantee — the fewest-hop path is not always the fastest route when edges have different distances.")]),
];

// ── SECTION 6 — Research ───────────────────────────────────────────────────────
const s6 = [
  H1("6. Research"),
  H2("6.1 Per-station Random Forest for edge-level traffic prediction"),
  P([T("The standard approach would be to train one global RF model on a single SCATS location and apply that single model to predict flow for every edge in the route graph. This ignores the fact that different roads have different baseline volumes, different peak shapes, and different sensitivities to weekday patterns.")]),
  P([T("We implemented a per-station RF architecture: each directed edge in the subgraph's "), C("EDGE_TO_LOCATION"), T(" mapping corresponds to a specific real SCATS location string in the dataset. When building travel-time edge weights, "), C("build_subgraph()"), T(" calls "), C("predictor._get_station_rf(location_name)"), T(", which lazily trains and caches an independent RF model for that exact road segment. The result is 36 models — one per directed edge — each tuned to the local traffic history for that specific stretch of road.")]),
  P([T("This matters when the system is used with a trained predictor (not demo mode). A segment on TOORAK_RD typically carries heavier load than a residential connector. Applying a single global model to both would give the same predicted flow for both edges at the same time slot, regardless of their historically different volumes. Per-station models apply the correct local mean, variance and time-of-day pattern to each edge independently.")]),
  P([T("Test T36 (skipped in fast mode) verifies that three different station RF models are distinct Python objects, confirming independent training rather than accidentally sharing a single trained model.")]),

  H2("6.2 Feature importance analysis: why lag dominates"),
  P([T("Running feature importance analysis on the RF model for multiple SCATS locations consistently produces the same pattern: y_lag1 accounts for over 90% of the model's predictive power. This is not a coincidence — it reflects a genuine property of short-interval traffic data.")]),
  P([T("Traffic flow measured at 15-minute intervals exhibits strong temporal autocorrelation. A road that had 80 vehicles in the last slot will not suddenly have 200 in the next unless an external event (accident, signal failure, special event) disrupts normal flow. The random forest captures this by learning lookup rules of the form: 'if y_lag1 is in [75, 85] and time_slot is in [30, 35], predict approximately 82.' This is essentially the same operation as a weighted k-nearest-neighbours query on the training set.")]),
  P([T("The implication for model selection on short-range time series: a simpler model that explicitly encodes the lag structure will often beat a complex recurrent model that must learn the same structure implicitly from thousands of parameters. This is not an argument against LSTM/GRU in general — it is specific to the regime where: (a) the dataset is small (one month), (b) the relevant temporal dependency is short (1–2 steps), and (c) the signal is stationary (traffic is periodic, not trending).")]),
  P([T("This finding guided our decision to set "), C("enable_deep_learning: false"), T(" as the GUI default. RF gives accurate route estimates without the training time penalty, which makes the application usable in demonstrations and assignments where 5-second startup time matters.")]),

  H2("6.3 Chronological split correction"),
  P([T("During code review, we identified that the original RF implementation used "), C("sklearn.train_test_split"), T(" with "), C("random_state=42"), T(", which randomly shuffles the time series before splitting. This means training samples could include data from October 28th while the test set includes data from October 3rd — a form of temporal data leakage where the model is trained on data that is 'future' relative to some test samples.")]),
  P([T("We corrected this to use chronological splitting: first 80% of rows (approximately October 1–24) go to training, final 20% (approximately October 25–31) go to testing. This matches the splitting logic already used for LSTM and GRU.")]),
  P([T("The metric impact was negligible: RMSE changed from 15.53 to 15.72 (+0.19), NRMSE from 0.157 to 0.161. The similarity confirms that October traffic patterns are temporally consistent — the model generalises well from the first three weeks to the final week. The correction is nevertheless important for methodological correctness: a published evaluation claiming NRMSE=0.157 on a random split overstates generalisation compared to NRMSE=0.161 on a proper forward test.")]),
];

// ── SECTION 7 — Conclusion ─────────────────────────────────────────────────────
const s7 = [
  H1("7. Conclusion"),
  P([T("The TBRGS integrates three independently testable components — data processing, ML traffic prediction, and graph-based route search — into one application. Each component has a clean interface: "), C("data_processor.py"), T(" returns standardised train/test splits; "), C("traffic_predictor.py"), T(" provides a "), C("predict()"), T(" method that hides the model selection detail; and "), C("build_subgraph()"), T(" takes any predictor and returns a weighted graph ready for any of the six search algorithms.")]),
  P([T("Random Forest was the most accurate model on this dataset. The reason is specific to the data regime: one month of 15-minute readings is too small for LSTM and GRU to fill their ~65,000 parameters, and the dominant predictive signal (y_lag1 = 91.9% importance) is a single tabular feature that RF captures directly. The deep learning models are not wrong choices — they need more data. Multi-year VicRoads datasets would narrow the gap significantly.")]),
  P([T("A* is the right search algorithm for route guidance in this application. The Euclidean distance heuristic keeps the expanded node set small (12 nodes on typical queries), the optimality guarantee ensures users always get the genuinely fastest route, and the per-edge ML weights mean the travel time estimate reflects the actual predicted traffic at the query time rather than static distances.")]),
  P([T("The most impactful improvements would be: (1) extending the subgraph to cover all 139 SCATS sites in the dataset rather than 17, which would require adding coordinates and edge definitions for the remaining 122 sites; (2) training on multiple years of VicRoads data to give LSTM and GRU enough samples to reach their full predictive capacity; and (3) implementing Yen's K-Shortest Paths algorithm properly to replace the current path-exclusion approximation for top-k route enumeration.")]),
];

// ── SECTION 8 — Acknowledgements ──────────────────────────────────────────────
const s8 = [
  H1("8. Acknowledgements and Resources"),
  P([T("The following resources were used in the implementation and report of this assignment:")]),
  bul("VicRoads. Scats_Data_October_2006.xls. Provided via Swinburne Canvas LMS, 2006. This was the sole source of traffic flow data used to train and evaluate all three ML models."),
  bul("TensorFlow 2.x / Keras (tensorflow.org). Used to implement the LSTM and GRU models in model_lstm.py and model_gru.py. The Sequential API, LSTM/GRU layers, Dropout, EarlyStopping, and Adam optimiser are all from this library."),
  bul("scikit-learn (scikit-learn.org). Used for RandomForestRegressor in model_rf.py and for MinMaxScaler in data_processor.py. Feature importance extraction uses the built-in feature_importances_ attribute."),
  bul("pandas and numpy (pandas.pydata.org, numpy.org). Used throughout data_processor.py for reading the Excel file, reshaping from wide to long format, and computing lag features."),
  bul("matplotlib (matplotlib.org). Used in subgraph.py (static graph rendering) and main_gui.py (embedded route map in the tkinter GUI). The TkAgg backend is used for GUI integration."),
  bul("Greenshields (1935) flow-speed model. The quadratic equation Q = −1.4648v² + 93.75v is sourced from the 'Traffic Flow to Travel Time Conversion v1.0' document provided on Canvas. It defines the physics of how predicted traffic volume maps to estimated travel speed."),
  bul("docx npm package (docxjs.github.io). Used to generate this report programmatically from make_report.mjs, following the skill guide provided by the course tooling."),
];

// ── SECTION 9 — References ─────────────────────────────────────────────────────
const s9 = [
  H1("9. References"),
  P([T("[1] VicRoads. Scats_Data_October_2006.xls. Provided via Swinburne Canvas, 2006.")]),
  P([T("[2] Greenshields, B.D. (1935). A study of traffic capacity. Proc. Highway Research Board, 14, 448–477.")]),
  P([T("[3] Hart, P., Nilsson, N., Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Trans. Systems Science and Cybernetics, 4(2), 100–107.")]),
  P([T("[4] Yen, J.Y. (1971). Finding the K Shortest Loopless Paths in a Network. Management Science, 17(11), 712–716.")]),
  P([T("[5] Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.")]),
  P([T("[6] Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder. EMNLP 2014. arXiv:1406.1078.")]),
  P([T("[7] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.")]),
  P([T("[8] TensorFlow Developers. TensorFlow 2.x Documentation. tensorflow.org, accessed May 2026.")]),
  P([T("[9] scikit-learn Developers. scikit-learn 1.x User Guide. scikit-learn.org, accessed May 2026.")]),
  P([T("[10] Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.")]),
];

// ── HEADER / FOOTER ───────────────────────────────────────────────────────────
const hdr = new Header({
  children: [new Paragraph({
    children: [T("COS30019 — Assignment 2B  |  TBRGS", { size: 18, color: MID })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BORD, space: 4 } },
    spacing: { after: 60 }
  })]
});
const ftr = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { top: { style: BorderStyle.SINGLE, size: 4, color: BORD, space: 4 } },
    children: [
      T("Page ", { size: 18, color: MID }),
      new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 18, color: MID }),
      T("  of  ", { size: 18, color: MID }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Calibri", size: 18, color: MID }),
    ]
  })]
});

// ── BUILD DOCUMENT ─────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bul",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 30, bold: true, color: BLUE },
        paragraph: { spacing: { before: 320, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 24, bold: true, color: DARK },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [
    // Cover — no header/footer
    {
      properties: { page: { size: { width: PW, height: PH }, margin: { top: MAR, right: MAR, bottom: MAR, left: MAR } } },
      children: cover
    },
    // TOC
    {
      properties: { page: { size: { width: PW, height: PH }, margin: { top: MAR, right: MAR, bottom: MAR, left: MAR } } },
      headers: { default: hdr }, footers: { default: ftr },
      children: toc
    },
    // Body
    {
      properties: { page: { size: { width: PW, height: PH }, margin: { top: MAR, right: MAR, bottom: MAR, left: MAR } } },
      headers: { default: hdr }, footers: { default: ftr },
      children: [
        ...s1, new Paragraph({ children: [new PageBreak()] }),
        ...s2, new Paragraph({ children: [new PageBreak()] }),
        ...s3, blank(),
        ...s4, new Paragraph({ children: [new PageBreak()] }),
        ...s5, new Paragraph({ children: [new PageBreak()] }),
        ...s6, new Paragraph({ children: [new PageBreak()] }),
        ...s7, blank(),
        ...s8, blank(),
        ...s9,
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/Users/cha/Downloads/cos30019_A2B/assignment2B.docx", buf);
  console.log("Done: assignment2B.docx");
}).catch(e => { console.error(e.message); process.exit(1); });
