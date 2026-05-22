import os
import sys
import json
import math
import warnings
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), ".matplotlib-cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(os.path.dirname(__file__), ".cache"),
)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

import data_processor as dp
from subgraph import SUBGRAPH_COORDS, SUBGRAPH_EDGES, EDGE_TO_LOCATION, build_subgraph
from search_algorithms import run_all_algorithms, ALGORITHM_LABELS
from tbrgs_graph import top_k_paths

# Load defaults from config file
_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(_CFG_PATH) as _f:
        _CFG = json.load(_f)
except Exception:
    _CFG = {}

# Colour palette — light theme
BG     = "#FFFFFF"
BG2    = "#F5F7FA"
BG3    = "#EDF0F5"
ACCENT = "#1565C0"
GREEN  = "#2E7D32"
AMBER  = "#E65100"
RED    = "#C62828"
TEXT   = "#1A202C"
TEXT2  = "#546E7A"
BORDER = "#CBD5E0"

# Nodes, dates, times
NODE_LIST = sorted(SUBGRAPH_COORDS.keys())

DATE_LIST   = []
DATE_TO_MDY = {}
for _d in range(1, 32):
    _disp = f"{_d:02d} Oct 2006"
    DATE_LIST.append(_disp)
    DATE_TO_MDY[_disp] = f"10/{_d:02d}/2006"

TIME_TO_SLOT = {}
for _h in range(24):
    for _m in (0, 15, 30, 45):
        TIME_TO_SLOT[f"{_h:02d}:{_m:02d}"] = (_h * 60 + _m) // 15
TIME_LIST = list(TIME_TO_SLOT.keys())

SEGMENT_LIST = []
SEGMENT_INFO = {}
for (_f, _t), _loc in sorted(EDGE_TO_LOCATION.items()):
    _disp = f"[{_f}->{_t}]  {_loc}"
    SEGMENT_LIST.append(_disp)
    SEGMENT_INFO[_disp] = (_f, _t, _loc)

# Defaults (from config or hardcoded fallback)
TRAIN_LOC_INDEX = _CFG.get("train_loc_index", 1)
TRAIN_EPOCHS    = _CFG.get("train_epochs", 30)
ENABLE_DEEP_LEARNING = _CFG.get("enable_deep_learning", True)
TOP_K           = _CFG.get("top_k_routes", 5)
DEFAULT_DATE    = _CFG.get("default_date", "16 Oct 2006")
DEFAULT_TIME    = _CFG.get("default_time", "08:00")
DEFAULT_START   = _CFG.get("default_start", NODE_LIST[0])
DEFAULT_END     = _CFG.get("default_end", NODE_LIST[-1])
WINDOW_SIZE     = _CFG.get("window_size", "1280x860")

# Make sure config defaults for start/end are valid nodes
if DEFAULT_START not in NODE_LIST:
    DEFAULT_START = NODE_LIST[0]
if DEFAULT_END not in NODE_LIST:
    DEFAULT_END = NODE_LIST[-1]


def _lbl(parent, text, fg=None, size=10, bold=False):
    return tk.Label(parent, text=text, bg=parent["bg"], fg=fg or TEXT2,
                    font=("Segoe UI", size, "bold" if bold else "normal"))


def _combo(parent, values, var, width=18):
    return ttk.Combobox(parent, values=values, textvariable=var,
                        width=width, state="readonly", font=("Segoe UI", 10))


def _btn(parent, text, cmd, color=None, fg=TEXT, size=10):
    color = color or ACCENT
    return tk.Button(parent, text=text, command=cmd, bg=color, fg=fg,
                     font=("Segoe UI", size, "bold"), relief=tk.FLAT,
                     activebackground=color, activeforeground=fg,
                     cursor="hand2", padx=14, pady=8, bd=1,
                     highlightthickness=1, highlightbackground=BORDER,
                     disabledforeground=TEXT2)


class TBRGSApp:
    def __init__(self, root):
        self.root = root
        root.title("TBRGS — Traffic-Based Route Guidance System")
        root.configure(bg=BG)
        root.geometry(WINDOW_SIZE)
        root.minsize(1050, 720)

        self.predictor       = None
        self.trained         = False
        self._top5_routes    = []
        self._top5_graph     = {}
        self._route_btns     = []
        self._selected_route = 0

        # Tab 3 inputs
        self.var_start = tk.StringVar(value=DEFAULT_START)
        self.var_end   = tk.StringVar(value=DEFAULT_END)
        self.var_date  = tk.StringVar(value=DEFAULT_DATE)
        self.var_time  = tk.StringVar(value=DEFAULT_TIME)

        # Tab 2 inputs
        self.var_seg   = tk.StringVar(value=SEGMENT_LIST[0])
        self.var_pdate = tk.StringVar(value=DEFAULT_DATE)
        self.var_ptime = tk.StringVar(value=DEFAULT_TIME)

        self._build_ui()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    def _build_ui(self):
        # Header bar
        hdr = tk.Frame(self.root, bg=BG, pady=10)
        hdr.pack(fill=tk.X, padx=20)
        tk.Label(hdr, text="TBRGS", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 20, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr, text="  Traffic-Based Route Guidance System",
                 bg=BG, fg=TEXT, font=("Segoe UI", 11)).pack(side=tk.LEFT)
        self.status_lbl = tk.Label(hdr, text="Not trained",
                                   bg=BG, fg=TEXT2, font=("Segoe UI", 10))
        self.status_lbl.pack(side=tk.RIGHT)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        # Style notebook
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=BG2, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG3, foreground=TEXT2,
                        padding=(16, 7), font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])
        style.configure("TCombobox", fieldbackground="white", background="white",
                        foreground=TEXT, selectbackground="white", selectforeground=TEXT)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)
        self.nb = nb

        self.tab_setup   = tk.Frame(nb, bg=BG)
        self.tab_predict = tk.Frame(nb, bg=BG)
        self.tab_route   = tk.Frame(nb, bg=BG)
        nb.add(self.tab_setup,   text="  1. Setup & Training  ")
        nb.add(self.tab_predict, text="  2. Traffic Prediction  ")
        nb.add(self.tab_route,   text="  3. Route Finder  ")

        self._build_setup_tab()
        self._build_predict_tab()
        self._build_route_tab()

    # -----------------------------------------------------------------------
    # Tab 1 — Setup & Training
    # -----------------------------------------------------------------------
    def _build_setup_tab(self):
        t = self.tab_setup
        P = dict(padx=24)

        tk.Label(t, text="Load dataset and train ML models",
                 bg=BG, fg=TEXT, font=("Segoe UI", 13, "bold")
                 ).pack(anchor=tk.W, pady=(20, 4), **P)
        tk.Label(t,
                 text="Train LSTM, GRU and Random Forest on the Boroondara SCATS "
                      "dataset (October 2006). Do this once before using Tab 3.",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 10),
                 justify=tk.LEFT, wraplength=900).pack(anchor=tk.W, pady=(0, 14), **P)

        row = tk.Frame(t, bg=BG)
        row.pack(anchor=tk.W, pady=(0, 6), **P)
        self.btn_load  = _btn(row, "Load Dataset", self._on_load, color=BG3, fg=TEXT)
        self.btn_load.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_train = _btn(row, "Train Models", self._on_train)
        self.btn_train.pack(side=tk.LEFT)

        self.setup_hint = tk.Label(
            t, text="Place 'Scats_Data_October_2006.xls' in the same folder, then click Load.",
            bg=BG, fg=TEXT2, font=("Segoe UI", 9))
        self.setup_hint.pack(anchor=tk.W, pady=(4, 14), **P)

        tk.Label(t, text="Model evaluation — RMSE / NRMSE / MAE / R²",
                 bg=BG, fg=TEXT, font=("Segoe UI", 11, "bold")
                 ).pack(anchor=tk.W, pady=(4, 8), **P)
        self.metrics_frame = tk.Frame(t, bg=BG)
        self.metrics_frame.pack(anchor=tk.W, fill=tk.X, **P)
        self._render_metric_cards(None)

        tk.Label(t, text="Training log",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 9, "bold")
                 ).pack(anchor=tk.W, pady=(16, 4), **P)
        self.setup_log = scrolledtext.ScrolledText(
            t, height=8, bg=BG3, fg=TEXT, font=("Consolas", 9),
            relief=tk.FLAT, wrap=tk.WORD, bd=0, padx=10, pady=8)
        self.setup_log.pack(fill=tk.BOTH, expand=True, pady=(0, 18), **P)
        self.setup_log.config(state=tk.DISABLED)

    def _render_metric_cards(self, comparison):
        for w in self.metrics_frame.winfo_children():
            w.destroy()

        best_name = None
        rows = {"LSTM": None, "GRU": None, "Random Forest": None}
        if comparison is not None:
            best_name = comparison["best_model"]
            for m in comparison["summary"]:
                rows[m["model"]] = m

        for i, (key, title) in enumerate([
            ("LSTM", "LSTM"), ("GRU", "GRU"), ("Random Forest", "Random Forest")
        ]):
            m       = rows.get(key)
            is_best = (key == best_name)
            card = tk.Frame(self.metrics_frame, bg=BG3,
                            highlightbackground=GREEN if is_best else BORDER,
                            highlightthickness=2 if is_best else 1)
            card.grid(row=0, column=i, padx=(0, 10), pady=2, sticky="nsew")
            self.metrics_frame.columnconfigure(i, weight=1)

            head = title + ("  ★ best" if is_best else "")
            tk.Label(card, text=head, bg=BG3,
                     fg=GREEN if is_best else TEXT,
                     font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=12, pady=(10, 6))
            if m is None:
                tk.Label(card, text="not trained yet", bg=BG3, fg=TEXT2,
                         font=("Segoe UI", 9)).pack(anchor=tk.W, padx=12, pady=(0, 12))
            else:
                for label, val in [
                    ("RMSE",  f"{m['rmse']:.2f}"),
                    ("NRMSE", f"{m['nrmse']:.3f}"),
                    ("MAE",   f"{m.get('mae', 0):.2f}"),
                    ("R²",    f"{m.get('r2', 0):.3f}"),
                ]:
                    row = tk.Frame(card, bg=BG3)
                    row.pack(fill=tk.X, padx=12, pady=1)
                    tk.Label(row, text=label, bg=BG3, fg=TEXT2,
                             font=("Segoe UI", 9), width=6, anchor=tk.W).pack(side=tk.LEFT)
                    tk.Label(row, text=val, bg=BG3, fg=TEXT,
                             font=("Consolas", 10)).pack(side=tk.LEFT)
                tk.Label(card, text="", bg=BG3).pack(pady=(0, 6))

        # Best model summary card
        card = tk.Frame(self.metrics_frame, bg=BG3,
                        highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=3, padx=0, pady=2, sticky="nsew")
        self.metrics_frame.columnconfigure(3, weight=1)
        tk.Label(card, text="Best model", bg=BG3, fg=TEXT2,
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=12, pady=(10, 6))
        if comparison is None:
            tk.Label(card, text="—", bg=BG3, fg=TEXT2,
                     font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, padx=12, pady=(0, 14))
        else:
            tk.Label(card, text=best_name, bg=BG3, fg=GREEN,
                     font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, padx=12, pady=(0, 2))
            tk.Label(card, text=f"NRMSE {comparison['best_nrmse']:.3f}",
                     bg=BG3, fg=TEXT2, font=("Segoe UI", 9)
                     ).pack(anchor=tk.W, padx=12, pady=(0, 14))

    # -----------------------------------------------------------------------
    # Tab 2 — Traffic Prediction
    # -----------------------------------------------------------------------
    def _build_predict_tab(self):
        t = self.tab_predict

        left = tk.Frame(t, bg=BG, width=330)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(24, 10), pady=20)
        left.pack_propagate(False)

        tk.Label(left, text="Traffic flow prediction", bg=BG, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        tk.Label(left,
                 text="Choose a road segment, date and time. All three trained "
                      "models output a predicted vehicle count (veh/15 min).",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 9),
                 justify=tk.LEFT, wraplength=300).pack(anchor=tk.W, pady=(2, 14))

        _lbl(left, "Road segment").pack(anchor=tk.W)
        _combo(left, SEGMENT_LIST, self.var_seg, width=46).pack(anchor=tk.W, pady=(2, 10))
        _lbl(left, "Date").pack(anchor=tk.W)
        _combo(left, DATE_LIST, self.var_pdate, width=46).pack(anchor=tk.W, pady=(2, 10))
        _lbl(left, "Time (15-min slot)").pack(anchor=tk.W)
        _combo(left, TIME_LIST, self.var_ptime, width=46).pack(anchor=tk.W, pady=(2, 12))

        self.btn_predict = _btn(left, "Predict Traffic Flow", self._on_predict)
        self.btn_predict.pack(fill=tk.X, pady=(4, 0))

        right = tk.Frame(t, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 24), pady=20)

        tk.Label(right, text="Prediction results",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.predict_box = scrolledtext.ScrolledText(
            right, height=20, bg=BG3, fg=TEXT, font=("Consolas", 10),
            relief=tk.FLAT, wrap=tk.WORD, bd=0, padx=12, pady=10)
        self.predict_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.predict_box.tag_config("head", foreground=ACCENT,
                                    font=("Consolas", 11, "bold"))
        self.predict_box.tag_config("best", foreground=GREEN,
                                    font=("Consolas", 11, "bold"))
        self.predict_box.tag_config("val",  foreground=TEXT,
                                    font=("Consolas", 11))
        self.predict_box.tag_config("meta", foreground=TEXT2)
        self.predict_box.config(state=tk.DISABLED)
        self._write_predict(
            "Train models on Tab 1, then select a road segment and click "
            "'Predict Traffic Flow'.\n", "meta")

    def _write_predict(self, text, tag="meta"):
        self.predict_box.config(state=tk.NORMAL)
        self.predict_box.insert(tk.END, text, tag)
        self.predict_box.see(tk.END)
        self.predict_box.config(state=tk.DISABLED)

    def _on_predict(self):
        if not self.trained:
            messagebox.showwarning("Not trained",
                                   "Train the models on Tab 1 first.")
            self.nb.select(self.tab_setup)
            return

        seg_disp    = self.var_seg.get()
        f_node, t_node, loc = SEGMENT_INFO[seg_disp]
        predict_day = DATE_TO_MDY[self.var_pdate.get()]
        time_slot   = TIME_TO_SLOT[self.var_ptime.get()]

        self.btn_predict.config(state=tk.DISABLED, text="Predicting...")
        self.root.update_idletasks()

        rows = []
        for key, name in (("lstm", "LSTM"), ("gru", "GRU"), ("rf", "Random Forest")):
            try:
                flow = self.predictor.predict(predict_day, time_slot, key,
                                              location_name=loc)
                rows.append((name, flow))
            except Exception:
                rows.append((name, None))

        best_name = self.predictor.comparison["best_model"]
        date_disp = self.var_pdate.get()
        time_disp = self.var_ptime.get()

        self.predict_box.config(state=tk.NORMAL)
        self.predict_box.delete("1.0", tk.END)
        self.predict_box.config(state=tk.DISABLED)

        self._write_predict("PREDICTION QUERY\n", "head")
        self._write_predict(f"  Segment  : {f_node} -> {t_node}\n", "val")
        self._write_predict(f"  Location : {loc}\n", "val")
        self._write_predict(f"  Date     : {date_disp}  {time_disp}\n\n", "val")
        self._write_predict("PREDICTED FLOW  (vehicles / 15 min)\n", "head")
        for name, flow in rows:
            is_best = (name == best_name)
            if flow is None:
                self._write_predict(f"  {name:<16} unavailable\n", "meta")
            else:
                mark = "  <- best model" if is_best else ""
                self._write_predict(f"  {name:<16} {flow:6.0f} veh/15min{mark}\n",
                                    "best" if is_best else "val")

        self.btn_predict.config(state=tk.NORMAL, text="Predict Traffic Flow")

    # -----------------------------------------------------------------------
    # Tab 3 — Route Finder
    # -----------------------------------------------------------------------
    def _build_route_tab(self):
        t = self.tab_route

        # Left input panel
        left = tk.Frame(t, bg=BG, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(24, 10), pady=20)
        left.pack_propagate(False)

        tk.Label(left, text="Route query", bg=BG, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 12))

        for text, var, values in (
            ("Start node", self.var_start, NODE_LIST),
            ("End node",   self.var_end,   NODE_LIST),
            ("Date",       self.var_date,  DATE_LIST),
            ("Time",       self.var_time,  TIME_LIST),
        ):
            _lbl(left, text).pack(anchor=tk.W)
            _combo(left, values, var, width=26).pack(anchor=tk.W, pady=(2, 10))

        self.btn_route = _btn(left, "Find Route", self._on_find_route)
        self.btn_route.pack(fill=tk.X, pady=(6, 8))

        self.route_hint = tk.Label(left, text="", bg=BG, fg=TEXT2,
                                   font=("Segoe UI", 9), justify=tk.LEFT,
                                   wraplength=230)
        self.route_hint.pack(anchor=tk.W)

        # Top-5 route buttons
        tk.Label(left, text=f"Top-{TOP_K} routes (click to view)",
                 bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")
                 ).pack(anchor=tk.W, pady=(18, 4))
        self.routes_frame = tk.Frame(left, bg=BG)
        self.routes_frame.pack(fill=tk.X)

        # Right panel — graph and report tabs
        right = tk.Frame(t, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                   padx=(0, 24), pady=20)

        self.route_view = ttk.Notebook(right)
        self.route_view.pack(fill=tk.BOTH, expand=True)
        self.map_tab = tk.Frame(self.route_view, bg=BG)
        self.report_tab = tk.Frame(self.route_view, bg=BG)
        self.route_view.add(self.map_tab, text="  Map  ")
        self.route_view.add(self.report_tab, text="  Details  ")

        tk.Label(self.map_tab, text="Pathfinding map — selected route highlighted",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)

        self.graph_fig = plt.Figure(figsize=(8.4, 5.8), facecolor="white")
        self.graph_ax  = self.graph_fig.add_subplot(111)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=self.map_tab)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                               pady=(4, 0))
        self._draw_empty_graph()

        tk.Label(self.report_tab, text="Route details and algorithm comparison",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.results_box = scrolledtext.ScrolledText(
            self.report_tab, height=10, bg=BG3, fg=TEXT, font=("Consolas", 10),
            relief=tk.FLAT, wrap=tk.WORD, bd=0, padx=10, pady=8)
        self.results_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.results_box.tag_config("head", foreground=ACCENT,
                                    font=("Consolas", 10, "bold"))
        self.results_box.tag_config("best", foreground=GREEN,
                                    font=("Consolas", 9, "bold"))
        self.results_box.tag_config("meta", foreground=TEXT2)
        self.results_box.config(state=tk.DISABLED)
        self._write_results("Train models on Tab 1, then run a route query.\n", "meta")

    # -----------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------
    def _log_setup(self, text):
        self.setup_log.config(state=tk.NORMAL)
        self.setup_log.insert(tk.END, text)
        self.setup_log.see(tk.END)
        self.setup_log.config(state=tk.DISABLED)

    def _write_results(self, text, tag="meta"):
        self.results_box.config(state=tk.NORMAL)
        self.results_box.insert(tk.END, text, tag)
        self.results_box.see(tk.END)
        self.results_box.config(state=tk.DISABLED)

    def _clear_results(self):
        self.results_box.config(state=tk.NORMAL)
        self.results_box.delete("1.0", tk.END)
        self.results_box.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Tab 1 actions
    # -----------------------------------------------------------------------
    def _on_load(self):
        try:
            df     = dp.load_raw()
            n_locs = len(dp.get_locations())
            self._log_setup(
                f"Dataset loaded: {len(df)} rows, {n_locs} SCATS locations.\n"
                f"Date range: October 2006 (31 days).\n\n")
            self.setup_hint.config(
                text="Dataset loaded. Click 'Train Models' to continue.", fg=GREEN)
            self.btn_load.config(text="Dataset Loaded", bg=BG3)
        except FileNotFoundError:
            messagebox.showerror(
                "File not found",
                "Could not find 'Scats_Data_October_2006.xls'.\n"
                "Place the file in the same folder as this program.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _on_train(self):
        self.btn_train.config(state=tk.DISABLED, text="Training...")
        self.btn_load.config(state=tk.DISABLED)
        self.status_lbl.config(text="Training...", fg=AMBER)
        if ENABLE_DEEP_LEARNING:
            self._log_setup("Training LSTM, GRU and Random Forest...\n")
        else:
            self._log_setup("Training Random Forest (LSTM/GRU disabled for this machine)...\n")
        if ENABLE_DEEP_LEARNING:
            self._log_setup(
                "Launching TensorFlow worker from the GUI main thread for macOS stability. "
                "The window may pause until training finishes.\n")
            self.root.after(100, lambda: self._train_worker(TRAIN_EPOCHS))
        else:
            threading.Thread(target=self._train_worker,
                             args=(TRAIN_EPOCHS,), daemon=True).start()

    def _train_worker(self, epochs):
        try:
            from traffic_predictor import TrafficPredictor

            predictor = TrafficPredictor()
            predictor.train_all(loc_index=TRAIN_LOC_INDEX,
                                epochs=epochs, verbose=0,
                                include_deep_learning=ENABLE_DEEP_LEARNING,
                                strict_deep_learning=ENABLE_DEEP_LEARNING,
                                external_deep_learning=ENABLE_DEEP_LEARNING)
            station_locs = list(set(EDGE_TO_LOCATION.values()))
            predictor.train_station_rf_models(station_locs)
            self.predictor = predictor
            self.trained   = True
            self.root.after(0, self._train_done, None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, self._train_done, str(e))

    def _train_done(self, error):
        self.btn_train.config(state=tk.NORMAL, text="Train Models")
        self.btn_load.config(state=tk.NORMAL)
        if error:
            self.status_lbl.config(text="Training failed", fg=RED)
            self.setup_hint.config(text="Training failed — check the log.", fg=RED)
            self._log_setup(f"\nERROR: {error}\n")
            messagebox.showerror("Training failed", error)
            return
        cmp = self.predictor.comparison
        self.status_lbl.config(
            text=f"Trained — best: {cmp['best_model']}", fg=GREEN)
        self.setup_hint.config(
            text="Training complete. Go to Tab 3 to find routes.", fg=GREEN)
        self._render_metric_cards(cmp)
        self._log_setup("\nTraining complete.\n")
        for m in cmp["summary"]:
            self._log_setup(
                f"  {m['model']:<16} RMSE={m['rmse']:.2f}  "
                f"NRMSE={m['nrmse']:.3f}  MAE={m.get('mae',0):.2f}  "
                f"R2={m.get('r2',0):.3f}\n")
        self._log_setup(f"  Best model: {cmp['best_model']}\n")

    # -----------------------------------------------------------------------
    # Tab 3 actions
    # -----------------------------------------------------------------------
    def _on_find_route(self):
        if not self.trained:
            messagebox.showwarning("Not trained",
                                   "Train the models on Tab 1 first.")
            self.nb.select(self.tab_setup)
            return

        start = self.var_start.get()
        end   = self.var_end.get()
        if start == end:
            messagebox.showwarning("Invalid query",
                                   "Start and end nodes must be different.")
            return

        predict_day = DATE_TO_MDY[self.var_date.get()]
        time_slot   = TIME_TO_SLOT[self.var_time.get()]

        self.route_hint.config(text="Searching all algorithms...")
        self.btn_route.config(state=tk.DISABLED, text="Searching...")
        self.root.update_idletasks()

        try:
            # Build the subgraph with per-edge ML-predicted travel times
            graph = build_subgraph(
                self.predictor, predict_day=predict_day,
                time_slot=time_slot, model="best", per_edge_flow=True)

            # Top-k simple paths ranked by ML-derived travel-time weights.
            routes = top_k_paths(graph, start, end, k=TOP_K)

            # All 6 A2A algorithm results for comparison
            algo_results = run_all_algorithms(
                start, end, predictor=self.predictor,
                predict_day=predict_day, time_slot=time_slot, model="best")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Search failed", str(e))
            self.btn_route.config(state=tk.NORMAL, text="Find Route")
            self.route_hint.config(text="")
            return

        self._top5_routes    = routes
        self._top5_graph     = graph
        self._selected_route = 0

        self._rebuild_route_buttons(routes, start, end)
        self._populate_results_box(routes, algo_results, graph, start, end)

        if routes:
            self._draw_route_graph(graph, routes[0]["path"], start, end, 0)
        else:
            self._draw_empty_graph()

        self.route_view.select(self.map_tab)
        self.btn_route.config(state=tk.NORMAL, text="Find Route")
        self.route_hint.config(
            text=f"{len(routes)} route(s) found." if routes else "No route found.")

    def _rebuild_route_buttons(self, routes, start, end):
        for w in self.routes_frame.winfo_children():
            w.destroy()
        self._route_btns = []

        if not routes:
            tk.Label(self.routes_frame, text="No routes found.",
                     bg=BG, fg=RED, font=("Segoe UI", 9)).pack(anchor=tk.W)
            return

        for i, r in enumerate(routes):
            path_str = " → ".join(r["path"])
            label    = f"Route {r['rank']}  ({r['time_min']:.1f} min)\n{path_str}"
            selected = (i == 0)
            btn = tk.Button(
                self.routes_frame, text=label,
                bg=BG3 if selected else "white",
                fg=TEXT,
                font=("Segoe UI", 8, "bold" if selected else "normal"),
                relief=tk.FLAT,
                justify=tk.LEFT, anchor=tk.W,
                padx=8, pady=6, cursor="hand2", wraplength=220,
                bd=1, highlightthickness=2 if selected else 1,
                highlightbackground=ACCENT if selected else BORDER,
                command=lambda idx=i: self._select_route(idx))
            btn.pack(fill=tk.X, pady=(0, 3))
            self._route_btns.append(btn)

    def _select_route(self, idx):
        self._selected_route = idx
        for i, btn in enumerate(self._route_btns):
            btn.config(bg=BG3 if i == idx else "white",
                       fg=TEXT,
                       font=("Segoe UI", 8, "bold" if i == idx else "normal"),
                       highlightthickness=2 if i == idx else 1,
                       highlightbackground=ACCENT if i == idx else BORDER)
        r     = self._top5_routes[idx]
        start = self.var_start.get()
        end   = self.var_end.get()
        self._draw_route_graph(self._top5_graph, r["path"], start, end, idx)

    def _best_model_line(self):
        if not self.predictor or not self.predictor.comparison:
            return "Best ML model: not trained"
        cmp = self.predictor.comparison
        return (f"Best ML model: {cmp['best_model']}  "
                f"RMSE={cmp['best_rmse']:.2f}  "
                f"NRMSE={cmp['best_nrmse']:.3f}")

    def _edge_flow_rows(self, path, graph):
        predict_day = DATE_TO_MDY[self.var_date.get()]
        time_slot   = TIME_TO_SLOT[self.var_time.get()]
        rows = []
        for i in range(len(path) - 1):
            f_node, t_node = path[i], path[i + 1]
            loc = EDGE_TO_LOCATION.get((f_node, t_node), "")
            weight = next((w for nb, w in graph.get(f_node, []) if nb == t_node), None)
            flow = self.predictor.predict(predict_day, time_slot, "best",
                                          location_name=loc)
            rows.append({
                "step": i + 1,
                "edge": f"{f_node}->{t_node}",
                "from": f_node,
                "to": t_node,
                "location": loc,
                "flow": flow,
                "weight": weight,
            })
        return rows

    def _populate_results_box(self, routes, algo_results, graph, start, end):
        self._clear_results()
        date_disp = self.var_date.get()
        time_disp = self.var_time.get()

        self._write_results(
            "PATHFINDING VISUALISATION OUTPUT\n", "head")
        self._write_results(
            f"  Query        : {start} -> {end}   {date_disp}  {time_disp}\n", "meta")
        self._write_results(
            "  Prediction   : f(location, date, time) -> traffic flow\n", "meta")
        self._write_results(
            "  Edge weight  : predicted flow -> travel time in minutes\n", "meta")
        self._write_results(
            "  Objective    : minimize total path weight\n", "best")
        self._write_results(f"  {self._best_model_line()}\n", "meta")

        if not routes:
            self._write_results("No route found.\n", "meta")
        else:
            best = routes[0]
            self._write_results("\nBEST PATH\n", "head")
            self._write_results(
                f"  Path         : {' -> '.join(best['path'])}\n", "best")
            self._write_results(
                f"  Total weight : {best['time_min']:.2f} min\n", "best")

            self._write_results(f"\nTOP-{len(routes)} ROUTES (ML travel-time weights)\n",
                                "head")
            for r in routes:
                path_str = " -> ".join(r["path"])
                tag = "best" if r["rank"] == 1 else "meta"
                self._write_results(
                    f"  Route {r['rank']}: {path_str}  "
                    f"({r['time_min']:.2f} min)\n", tag)

        # Per-edge flow breakdown for the highlighted best route
        if routes:
            self._write_results("\nROUTE EDGE DETAILS — listed in path order\n", "head")
            self._write_results(
                f"  {'#':<3}{'Edge':<12}{'Location':<34}{'Flow':>8}{'Weight':>9}\n", "meta")
            self._write_results(f"  {'-'*70}\n", "meta")
            for row in self._edge_flow_rows(routes[0]["path"], graph):
                flow = row["flow"]
                flow_str = f"{flow:.0f}" if flow == flow else "n/a"
                weight = row["weight"]
                weight_str = f"{weight:.2f}m" if weight is not None else "n/a"
                self._write_results(
                    f"  {row['step']:<3}{row['edge']:<12}{row['location'][:32]:<34}"
                    f"{flow_str:>9}{weight_str:>9}\n", "meta")

        # Whiteboard requirement: compare 3 pathfinding algorithms.
        self._write_results("\n3 ALGORITHMS — compare total weight\n", "head")
        self._write_results(
            f"  {'Algorithm':<32}{'Weight':>10}{'Hops':>6}{'Path'}\n", "meta")
        self._write_results(f"  {'-'*78}\n", "meta")
        required_algos = {"astar", "bfs", "dfs"}
        required_results = [r for r in algo_results if r["algorithm"] in required_algos]
        best_required = next((r for r in required_results if r["found"]), None)
        for r in required_results:
            weight = f"{r['total_time_min']:.2f}m" if r["found"] else "no path"
            mark = "  <- lowest weight" if r is best_required else ""
            path = " -> ".join(r["path"]) if r["path"] else "-"
            tag = "best" if r is best_required else "meta"
            self._write_results(
                f"  {r['label']:<32}{weight:>10}{r['hops']:>6}  {path}{mark}\n",
                tag)

        self._write_results("\nALL 6 A2A ALGORITHMS — sorted by total travel time\n", "head")
        self._write_results(
            f"  {'Algorithm':<32}{'Time(min)':>10}{'Hops':>6}{'Nodes':>7}\n", "meta")
        self._write_results(f"  {'-'*55}\n", "meta")
        weight_unaware = {"bfs", "dfs", "cus1"}
        for i, r in enumerate(algo_results):
            tw   = f"{r['total_time_min']:.2f}" if r["found"] else "no path"
            mark = "  <- Optimal" if (i == 0 and r["found"]) else ""
            note = " (weight-unaware)" if r["algorithm"] in weight_unaware else ""
            tag  = "best" if (i == 0 and r["found"]) else "meta"
            self._write_results(
                f"  {r['label']:<32}{tw:>10}{r['hops']:>6}"
                f"{r['nodes_created']:>7}{mark}{note}\n", tag)
        self.results_box.yview_moveto(0)

    # -----------------------------------------------------------------------
    # Graph drawing
    # -----------------------------------------------------------------------
    def _draw_empty_graph(self):
        ax = self.graph_ax
        ax.clear()
        self.graph_fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Run a route query — the path will appear here.",
                ha="center", va="center", color=TEXT2,
                fontsize=11, transform=ax.transAxes)
        self.graph_canvas.draw()

    def _draw_route_graph(self, graph, path, origin, destination, route_idx=0):
        coords = SUBGRAPH_COORDS
        ax     = self.graph_ax
        ax.clear()
        self.graph_fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")

        lats = [v[0] for v in coords.values()]
        lons = [v[1] for v in coords.values()]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        def proj(nid, m=0.11):
            lat, lon = coords[nid][:2]
            x = m + (lon - lon_min) / (lon_max - lon_min + 1e-9) * (1 - 2 * m)
            y = m + (lat - lat_min) / (lat_max - lat_min + 1e-9) * (1 - 2 * m)
            return x, y

        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))
        edge_rows = {(r["from"], r["to"]): r for r in self._edge_flow_rows(path, graph)}

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.10, 1.08)

        # Draw edges
        drawn = set()
        for f, t, dist in SUBGRAPH_EDGES:
            x1, y1  = proj(f)
            x2, y2  = proj(t)
            is_path = (f, t) in path_edges
            pair    = frozenset([f, t])
            dx, dy  = x2 - x1, y2 - y1
            length  = math.sqrt(dx * dx + dy * dy) + 1e-9
            offset  = 0.006 if pair in drawn else 0.0
            px, py  = -dy / length * offset, dx / length * offset
            ux, uy  = dx / length, dy / length
            clear   = 0.026 if is_path else 0.014
            sx, sy  = x1 + ux * clear + px, y1 + uy * clear + py
            ex, ey  = x2 - ux * clear + px, y2 - uy * clear + py

            color   = AMBER if is_path else "#8EA1AA"
            lw      = 3.2 if is_path else 1.15
            alpha   = 1.0 if is_path else 0.68

            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, alpha=alpha,
                                        mutation_scale=10 if is_path else 7,
                                        shrinkA=0, shrinkB=0))

            drawn.add(pair)

        # Draw nodes
        for nid in coords:
            x, y    = proj(nid)
            in_path = nid in path
            if nid == origin:
                color, size, ring = GREEN,      700, 3.2
            elif nid == destination:
                color, size, ring = RED,        700, 3.2
            elif in_path:
                color, size, ring = AMBER,      620, 2.9
            else:
                color, size, ring = "#78909C",  150,  1.3

            ax.scatter(x, y, s=size, c=color, zorder=5,
                       edgecolors="white", linewidths=ring)
            ax.text(x, y, nid, ha="center", va="center", zorder=7,
                    fontsize=8.6 if in_path or nid in (origin, destination) else 6.2,
                    fontweight="bold", fontfamily="monospace",
                    color="white")

        patches = [
            mpatches.Patch(color=GREEN,    label=f"Start ({origin})"),
            mpatches.Patch(color=RED,      label=f"End ({destination})"),
            mpatches.Patch(color=AMBER,    label=f"Route {route_idx + 1} path"),
            mpatches.Patch(color="#78909C", label="Other roads/nodes"),
        ]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.05), ncol=4,
                  facecolor="white", edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=8, framealpha=0.95)

        title = f"Route {route_idx + 1}"
        if self._top5_routes and route_idx < len(self._top5_routes):
            r      = self._top5_routes[route_idx]
            title += f" | {origin} -> {destination} | total {r['time_min']:.1f} min"
        ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=6)

        if self._top5_routes and route_idx < len(self._top5_routes):
            route = self._top5_routes[route_idx]
            cmp = self.predictor.comparison if self.predictor else None
            best_model = cmp["best_model"] if cmp else "n/a"
            nrmse = f"{cmp['best_nrmse']:.3f}" if cmp else "n/a"
            date_disp = self.var_date.get()
            time_disp = self.var_time.get()
            summary = (
                f"{date_disp} {time_disp}\n"
                f"Route weight: {route['time_min']:.2f} min\n"
                f"Model: {best_model}\n"
                f"NRMSE: {nrmse}"
            )
            ax.text(0.02, 0.98, summary,
                    ha="left", va="top", transform=ax.transAxes,
                    fontsize=8.6, color=TEXT, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=BORDER, alpha=0.96, lw=0.9))

            path_nodes = " -> ".join(route["path"])
            if len(path_nodes) > 60:
                path_nodes = path_nodes[:57] + "..."
            ax.text(0.50, 0.02,
                    f"Selected path: {path_nodes}",
                    ha="center", va="bottom", transform=ax.transAxes,
                    fontsize=9, color=TEXT, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white",
                              ec=BORDER, alpha=0.96, lw=0.9))

        self.graph_fig.tight_layout()
        self.graph_canvas.draw()


# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TCombobox", fieldbackground="white", background="white",
                    foreground=TEXT, arrowcolor=TEXT2,
                    selectbackground="white", selectforeground=TEXT)
    root.option_add("*TCombobox*Listbox.background", "white")
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    TBRGSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
