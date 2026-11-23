"""Mistie Tool desktop application (tkinter + matplotlib). Runs fully offline."""
import json, os, sys, traceback, html
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from .session import Session
from .measure import apply_correction
from . import __version__

TOP_C, BASE_C, TIE_C = "#16b3c9", "#d9468f", "#ffd43b"
LINE_COLORS = ["#1f77b4", "#2ca02c", "#e377c2", "#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#17becf",
               "#bcbd22", "#7f7f7f", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"]
RECENT = os.path.join(os.path.expanduser("~"), ".mistie_tool_recent.json")
MAP_MODES = {"Tie status": None, "Mistie: time shift (ms)": "shift", "Residual: time (ms)": "res_shift",
             "Mistie: phase (deg)": "phase", "Residual: phase (deg)": "res_phase"}
KEYS_HELP = """Tie panel
  Drag the cyan (top) / magenta (base) line, or click to move the nearest one
  T / B         set top / base at the mouse position
  1 / 2 / 3     raw / tie-corrected / network-corrected view

Moving around
  N / P         next / previous tie (in table order)
  W             next worst tie (largest misfit first)
  Click a tie on the map or in the table

Editing
  A / O / X     use tie: auto / force on / exclude (applies to all selected ties)
  R             reset window of selected ties to the default
  Ctrl+Z / Ctrl+Y   undo / redo
  Ctrl+S        save project
  Right-click the tie or line tables for more actions
"""


class Busy:
    """Small progress window used while reading data."""
    def __init__(self, root, title):
        self.top = tk.Toplevel(root); self.top.title(title); self.top.transient(root); self.top.resizable(False, False)
        self.lbl = ttk.Label(self.top, text="Starting...", width=60); self.lbl.pack(padx=16, pady=(14, 6))
        self.bar = ttk.Progressbar(self.top, length=360, mode="determinate", maximum=1.0); self.bar.pack(padx=16, pady=(0, 14))
        self.top.update()

    def __call__(self, msg, frac=None):
        self.lbl.config(text=msg)
        if frac is not None:
            self.bar["value"] = frac
        self.top.update()

    def close(self):
        self.top.destroy()


class WindowDialog(tk.Toplevel):
    def __init__(self, root, title, w):
        super().__init__(root); self.title(title); self.transient(root); self.resizable(False, False); self.result = None
        f = ttk.Frame(self, padding=12); f.pack()
        ttk.Label(f, text="Top (s)").grid(row=0, column=0, sticky="w"); ttk.Label(f, text="Base (s)").grid(row=1, column=0, sticky="w")
        self.e0, self.e1 = ttk.Entry(f, width=10), ttk.Entry(f, width=10)
        self.e0.insert(0, f"{w[0]:.3f}"); self.e1.insert(0, f"{w[1]:.3f}")
        self.e0.grid(row=0, column=1, padx=6, pady=3); self.e1.grid(row=1, column=1, padx=6, pady=3)
        b = ttk.Frame(f); b.grid(row=2, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(b, text="OK", command=self.ok).pack(side="left", padx=4); ttk.Button(b, text="Cancel", command=self.destroy).pack(side="left")
        self.bind("<Return>", lambda e: self.ok()); self.e0.focus_set(); self.grab_set(); self.wait_window()

    def ok(self):
        try:
            self.result = (float(self.e0.get()), float(self.e1.get()))
        except ValueError:
            messagebox.showerror("Window", "Enter numbers in seconds", parent=self); return
        self.destroy()


class LockDialog(tk.Toplevel):
    def __init__(self, root, line, st, current):
        super().__init__(root); self.title(f"Correction for {line}"); self.transient(root); self.resizable(False, False); self.result = None
        f = ttk.Frame(self, padding=12); f.pack()
        self.lock = tk.BooleanVar(value=st["lock"])
        ttk.Checkbutton(f, text="Lock this line at the values below (it will not be solved)", variable=self.lock).grid(row=0, column=0, columnspan=3, sticky="w")
        self.ent = {}
        vals = st if st["lock"] else current
        for r, (k, lab) in enumerate([("shift", "Shift (ms)"), ("phase", "Phase (deg)"), ("gain", "Gain")], start=1):
            ttk.Label(f, text=lab).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(f, width=10); e.insert(0, f"{vals[k]:.3f}"); e.grid(row=r, column=1, padx=6, pady=2); self.ent[k] = e
        ttk.Label(f, text=f"Current solution: {current['shift']:+.2f} ms, {current['phase']:+.1f} deg, gain {current['gain']:.3f}",
                  foreground="#555").grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        b = ttk.Frame(f); b.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(b, text="OK", command=self.ok).pack(side="left", padx=4); ttk.Button(b, text="Cancel", command=self.destroy).pack(side="left")
        self.grab_set(); self.wait_window()

    def ok(self):
        try:
            v = {k: float(e.get()) for k, e in self.ent.items()}
        except ValueError:
            messagebox.showerror("Correction", "Enter numbers", parent=self); return
        if v["gain"] <= 0:
            messagebox.showerror("Correction", "Gain must be positive", parent=self); return
        self.result = (self.lock.get(), v); self.destroy()


class SettingsDialog(tk.Toplevel):
    FIELDS = [("min_corr", "Minimum correlation for auto ties"), ("max_half_diff_ms", "Max shallow/deep shift difference (ms)"),
              ("near_end_traces", "Near-line-end distance (traces)"), ("near_end_weight", "Weight of near-end ties"),
              ("reject_tol_ms", "Misfit tolerance, time (ms)"), ("reject_tol_deg", "Misfit tolerance, phase (deg)"),
              ("max_lag_ms", "Maximum shift searched (ms)"), ("tie_avg_traces", "Traces averaged at a tie")]

    def __init__(self, root, cfg):
        super().__init__(root); self.title("Settings"); self.transient(root); self.resizable(False, False); self.result = None
        f = ttk.Frame(self, padding=12); f.pack(); self.ent = {}
        for r, (k, lab) in enumerate(self.FIELDS):
            ttk.Label(f, text=lab).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(f, width=10); e.insert(0, str(cfg[k])); e.grid(row=r, column=1, padx=6, pady=2); self.ent[k] = e
        ttk.Label(f, text="Changing the last two re-measures every tie.", foreground="#555").grid(row=len(self.FIELDS), column=0, columnspan=2, sticky="w", pady=(6, 0))
        b = ttk.Frame(f); b.grid(row=len(self.FIELDS) + 1, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(b, text="OK", command=self.ok).pack(side="left", padx=4); ttk.Button(b, text="Cancel", command=self.destroy).pack(side="left")
        self.grab_set(); self.wait_window()

    def ok(self):
        try:
            self.result = {k: (int(float(e.get())) if k in ("near_end_traces", "tie_avg_traces") else float(e.get())) for k, e in self.ent.items()}
        except ValueError:
            messagebox.showerror("Settings", "Enter numbers", parent=self); return
        self.destroy()


class MistieApp:
    def __init__(self, root, path=None):
        self.root, self.S, self.cur = root, None, 0
        self.view = tk.StringVar(value="raw"); self.map_mode = tk.StringVar(value="Mistie: time shift (ms)")
        self.map_labels = tk.StringVar(value="misfits + selected"); self.gain = tk.DoubleVar(value=1.0)
        self.sort_col, self.sort_rev, self.drag, self.mouse_t, self.map_lims = None, False, None, None, None
        root.title("Mistie Tool"); root.geometry("1600x960"); root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._style(); self._menu(); self._layout(); self._keys()
        if path:
            root.after(100, lambda: self.open_path(path))
        else:
            self._empty_state()

    # ======================================================== construction
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("vista" if sys.platform == "win32" else "clam")
        except tk.TclError:
            pass
        st.configure("Treeview", rowheight=22)
        self.mono = ("Consolas", 10) if sys.platform == "win32" else ("DejaVu Sans Mono", 9)

    def _menu(self):
        m = tk.Menu(self.root); self.root.config(menu=m)
        fm = tk.Menu(m, tearoff=0); m.add_cascade(label="File", menu=fm)
        fm.add_command(label="New project from SEG-Y folder...", command=self.new_project)
        fm.add_command(label="Open project...", command=self.open_dialog, accelerator="Ctrl+O")
        self.recent_menu = tk.Menu(fm, tearoff=0); fm.add_cascade(label="Open recent", menu=self.recent_menu); self._recent_menu()
        fm.add_separator()
        fm.add_command(label="Save", command=self.save, accelerator="Ctrl+S"); fm.add_command(label="Save as...", command=self.save_as)
        fm.add_separator()
        fm.add_command(label="Export tables (CSV)...", command=self.export_csv)
        fm.add_command(label="Write corrected SEG-Y...", command=self.write_segy)
        fm.add_command(label="Export report (HTML)...", command=self.export_report)
        fm.add_separator(); fm.add_command(label="Exit", command=self.on_close)
        em = tk.Menu(m, tearoff=0); m.add_cascade(label="Edit", menu=em)
        em.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z"); em.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        em.add_separator()
        em.add_command(label="Set window for selected ties...", command=lambda: self.window_dialog("selected"))
        em.add_command(label="Set window for ALL ties...", command=lambda: self.window_dialog("all"))
        em.add_command(label="Copy current tie's window to selected ties", command=self.copy_window)
        em.add_command(label="Reset window of selected ties", command=self.reset_selected, accelerator="R")
        em.add_separator()
        for lab, mode, acc in [("Use selected ties: auto", "auto", "A"), ("Force selected ties on", "on", "O"), ("Exclude selected ties", "off", "X")]:
            em.add_command(label=lab, command=lambda md=mode: self.use_selected(md), accelerator=acc)
        em.add_separator(); em.add_command(label="Settings...", command=self.settings)
        vm = tk.Menu(m, tearoff=0); m.add_cascade(label="View", menu=vm)
        for lab, v in [("Raw", "raw"), ("Tie-corrected", "tie"), ("Network-corrected", "net")]:
            vm.add_radiobutton(label=f"Tie panel: {lab}", variable=self.view, value=v, command=self.draw_tie)
        vm.add_separator()
        for lab in MAP_MODES:
            vm.add_radiobutton(label=f"Map: {lab}", variable=self.map_mode, value=lab, command=self.draw_map)
        vm.add_separator()
        for lab in ("all", "misfits + selected", "selected only", "none"):
            vm.add_radiobutton(label=f"Map labels: {lab}", variable=self.map_labels, value=lab, command=self.draw_map)
        nm = tk.Menu(m, tearoff=0); m.add_cascade(label="Network", menu=nm)
        self.ref_menu = tk.Menu(nm, tearoff=0); nm.add_cascade(label="Reference line", menu=self.ref_menu)
        self.solve_by = tk.StringVar(value="line")
        nm.add_radiobutton(label="Solve one correction per line", variable=self.solve_by, value="line", command=self.set_solve_by)
        nm.add_radiobutton(label="Solve one correction per survey/vintage", variable=self.solve_by, value="survey", command=self.set_solve_by)
        nm.add_separator()
        nm.add_command(label="Exclude worst-fitting tie", command=self.reject_worst)
        nm.add_command(label="Auto-reject misfits", command=self.auto_reject)
        nm.add_command(label="Set all ties to auto", command=self.all_auto)
        nm.add_separator()
        nm.add_command(label="Save scenario...", command=self.save_scenario)
        self.scen_menu = tk.Menu(nm, tearoff=0); nm.add_cascade(label="Load scenario", menu=self.scen_menu)
        self.scen_del_menu = tk.Menu(nm, tearoff=0); nm.add_cascade(label="Delete scenario", menu=self.scen_del_menu)
        hm = tk.Menu(m, tearoff=0); m.add_cascade(label="Help", menu=hm)
        hm.add_command(label="Keys and mouse", command=lambda: messagebox.showinfo("Keys and mouse", KEYS_HELP))
        hm.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", f"Mistie Tool {__version__}\n2D seismic mistie analysis and network adjustment.\n"
                     "Runs locally; no data leaves this computer."))

    def _layout(self):
        outer = ttk.PanedWindow(self.root, orient="vertical"); outer.pack(fill="both", expand=True)
        main = ttk.PanedWindow(outer, orient="horizontal"); outer.add(main, weight=4)
        # ---- left: map + tie table
        left = ttk.PanedWindow(main, orient="vertical"); main.add(left, weight=2)
        mapf = ttk.Frame(left); left.add(mapf, weight=3)
        bar = ttk.Frame(mapf); bar.pack(fill="x")
        ttk.Label(bar, text="Map").pack(side="left", padx=(4, 6))
        cb = ttk.Combobox(bar, textvariable=self.map_mode, values=list(MAP_MODES), state="readonly", width=24); cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw_map())
        ttk.Button(bar, text="Fit", width=5, command=self.fit_map).pack(side="left", padx=4)
        self.map_fig = Figure(figsize=(4.6, 3.8), dpi=96); self.map_canvas = FigureCanvasTkAgg(self.map_fig, mapf)
        tbf = ttk.Frame(mapf); tbf.pack(side="bottom", fill="x")
        self.map_tb = NavigationToolbar2Tk(self.map_canvas, tbf, pack_toolbar=False); self.map_tb.update(); self.map_tb.pack(side="left")
        self.map_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.map_canvas.mpl_connect("button_press_event", self.on_map_click)
        tf = ttk.Frame(left); left.add(tf, weight=2)
        cols = ("tie", "window", "shift", "phase", "gain", "corr", "half", "use", "res_ms", "res_deg", "status")
        heads = ("Tie", "Window (s)", "Shift ms", "Phase", "Gain", "r", "Half diff", "Use", "Resid ms", "Resid deg", "Status")
        widths = (120, 82, 62, 52, 46, 42, 60, 46, 62, 62, 64)
        self.tt = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        for c, h, w in zip(cols, heads, widths):
            self.tt.heading(c, text=h, command=lambda c=c: self.sort_ties(c)); self.tt.column(c, width=w, anchor="e" if c not in ("tie", "use", "status") else "w", stretch=False)
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tt.yview); self.tt.configure(yscrollcommand=sb.set)
        sbx = ttk.Scrollbar(tf, orient="horizontal", command=self.tt.xview); self.tt.configure(xscrollcommand=sbx.set)
        self.tt.grid(row=0, column=0, sticky="nsew"); sb.grid(row=0, column=1, sticky="ns"); sbx.grid(row=1, column=0, sticky="ew")
        tf.rowconfigure(0, weight=1); tf.columnconfigure(0, weight=1)
        self.tt.tag_configure("misfit", foreground="#c92a2a"); self.tt.tag_configure("unused", foreground="#888888")
        self.tt.bind("<<TreeviewSelect>>", self.on_tie_select); self.tt.bind("<Button-3>", self.tie_menu)
        # ---- centre: tie panel
        cen = ttk.Frame(main); main.add(cen, weight=5)
        bar = ttk.Frame(cen); bar.pack(fill="x", pady=(2, 2))
        ttk.Button(bar, text="< Prev", command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(bar, text="Next >", command=lambda: self.step(1)).pack(side="left", padx=2)
        ttk.Button(bar, text="Next worst (W)", command=self.next_worst).pack(side="left", padx=(2, 12))
        for lab, v in [("Raw", "raw"), ("Tie-corrected", "tie"), ("Network-corrected", "net")]:
            ttk.Radiobutton(bar, text=lab, variable=self.view, value=v, command=self.draw_tie).pack(side="left", padx=3)
        ttk.Label(bar, text="   Gain").pack(side="left")
        ttk.Scale(bar, from_=0.3, to=4.0, variable=self.gain, command=lambda v: self.draw_tie(), length=120).pack(side="left", padx=4)
        self.tie_title = ttk.Label(cen, text="", font=("Segoe UI", 11, "bold") if sys.platform == "win32" else ("DejaVu Sans", 11, "bold"))
        self.tie_title.pack(anchor="w", padx=4)
        self.tie_fig = Figure(figsize=(8, 6), dpi=96); self.tie_canvas = FigureCanvasTkAgg(self.tie_fig, cen)
        self.tie_canvas.get_tk_widget().pack(fill="both", expand=True)
        for ev, fn in [("button_press_event", self.on_press), ("motion_notify_event", self.on_move), ("button_release_event", self.on_release)]:
            self.tie_canvas.mpl_connect(ev, fn)
        ttk.Label(cen, text="Drag the cyan top / magenta base lines (or press T / B at the mouse). Left of the yellow line: first line approaching the tie; right: second line leaving it.",
                  foreground="#555").pack(anchor="w", padx=4)
        # ---- right: overlay
        rt = ttk.Frame(main); main.add(rt, weight=2)
        self.ov_fig = Figure(figsize=(3.2, 6), dpi=96); self.ov_canvas = FigureCanvasTkAgg(self.ov_fig, rt)
        self.ov_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.meas = tk.Text(rt, height=9, width=38, font=self.mono, relief="flat", background=self.root.cget("background"))
        self.meas.pack(fill="x", padx=4, pady=4)
        # ---- bottom notebook
        nb = ttk.Notebook(outer); outer.add(nb, weight=1)
        lf = ttk.Frame(nb); nb.add(lf, text="Lines")
        lcols = ("line", "survey", "shift", "phase", "gain", "ties", "status")
        self.lt = ttk.Treeview(lf, columns=lcols, show="headings", selectmode="browse", height=6)
        for c, h, w in zip(lcols, ("Line", "Survey", "Shift ms", "Phase deg", "Gain", "Ties used", "Status"), (110, 110, 80, 80, 70, 80, 360)):
            self.lt.heading(c, text=h); self.lt.column(c, width=w, anchor="w" if c in ("line", "survey", "status") else "e", stretch=c == "status")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.lt.yview); self.lt.configure(yscrollcommand=sb.set)
        self.lt.pack(side="left", fill="both", expand=True); sb.pack(side="left", fill="y")
        self.lt.bind("<Double-1>", lambda e: self.edit_line()); self.lt.bind("<Button-3>", self.line_menu)
        lpf = ttk.Frame(nb); nb.add(lpf, text="Loops")
        self.lp = ttk.Treeview(lpf, columns=("loop", "ms", "deg", "status"), show="headings", height=6)
        for c, h, w in zip(("loop", "ms", "deg", "status"), ("Loop", "Closure ms", "Closure deg", "Status"), (520, 90, 90, 160)):
            self.lp.heading(c, text=h); self.lp.column(c, width=w, anchor="w" if c in ("loop", "status") else "e")
        self.lp.pack(fill="both", expand=True); self.lp.tag_configure("bad", foreground="#c92a2a")
        wf = ttk.Frame(nb); nb.add(wf, text="Warnings"); self.nb = nb
        self.warn = tk.Text(wf, height=6, font=self.mono, relief="flat", background=self.root.cget("background")); self.warn.pack(fill="both", expand=True)
        self.status = ttk.Label(self.root, text="", anchor="w", relief="sunken"); self.status.pack(fill="x", side="bottom")

    def _keys(self):
        r = self.root
        r.bind_all("<Control-z>", lambda e: self.undo()); r.bind_all("<Control-y>", lambda e: self.redo())
        r.bind_all("<Control-Shift-Z>", lambda e: self.redo())
        r.bind_all("<Control-Z>", lambda e: self.undo())  # Caps Lock on: still undo; r.bind_all("<Control-s>", lambda e: self.save())
        r.bind_all("<Control-o>", lambda e: self.open_dialog())
        keymap = {"n": lambda: self.step(1), "p": lambda: self.step(-1), "w": self.next_worst,
                  "a": lambda: self.use_selected("auto"), "o": lambda: self.use_selected("on"), "x": lambda: self.use_selected("off"),
                  "r": self.reset_selected, "t": lambda: self.key_pick(0), "b": lambda: self.key_pick(1),
                  "1": lambda: self.set_view("raw"), "2": lambda: self.set_view("tie"), "3": lambda: self.set_view("net")}

        def on_key(e):
            if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)) or e.state & 0x4:
                return
            f = keymap.get((e.char or "").lower())
            if f and self.S:
                f(); return "break"
        r.bind_all("<Key>", on_key)

    def _empty_state(self):
        self.tie_fig.clear(); ax = self.tie_fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.55, "Mistie Tool", ha="center", fontsize=22, fontweight="bold")
        ax.text(0.5, 0.42, "File > New project from SEG-Y folder...   or   File > Open project...", ha="center", fontsize=11, color="#555")
        self.tie_canvas.draw_idle(); self.status.config(text="No project open")

    # ======================================================== project files
    def _guard(self, fn):
        try:
            return fn()
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Mistie Tool", f"{type(ex).__name__}: {ex}")

    def _guard_ok(self, fn):
        """Like _guard, but True only when fn ran without an error (a failed save must not count as saved)."""
        try:
            fn()
            return True
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Mistie Tool", f"{type(ex).__name__}: {ex}")
            return False

    def confirm_discard(self):
        if self.S and self.S.dirty:
            r = messagebox.askyesnocancel("Unsaved changes", "Save changes to the current project first?")
            if r is None:
                return False
            if r:
                return self.save()
        return True

    def new_project(self):
        if not self.confirm_discard():
            return
        folder = filedialog.askdirectory(title="Folder with the 2D SEG-Y lines")
        if not folder:
            return
        files = Session.scan_folder(folder)
        if not files:
            messagebox.showerror("New project", "No .sgy / .segy files in that folder."); return
        nav = None
        while True:
            busy = Busy(self.root, "Reading SEG-Y")
            try:
                S = Session.from_folder(folder, files, nav_csv=nav, progress=busy)
                busy.close(); break
            except ValueError as ex:
                busy.close()
                if ("coordinates" in str(ex) or "no intersections" in str(ex)) and nav is None:
                    if not messagebox.askokcancel("Coordinates needed", f"{ex}\n\nChoose a navigation CSV (columns line,cdp,x,y)?"):
                        return
                    nav = filedialog.askopenfilename(title="Navigation CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
                    if not nav:
                        return
                    continue
                messagebox.showerror("New project", str(ex)); return
            except Exception as ex:
                busy.close(); traceback.print_exc(); messagebox.showerror("New project", f"{type(ex).__name__}: {ex}"); return
        self.set_session(S)
        messagebox.showinfo("New project", f"Loaded {len(S.names)} lines and found {len(S.P.ties)} ties.\n"
                                           f"Use File > Save to create the project file.")

    def open_dialog(self):
        if not self.confirm_discard():
            return
        p = filedialog.askopenfilename(title="Open project", filetypes=[("Mistie projects", "*.mistie.json *.json"), ("All", "*.*")])
        if p:
            self.open_path(p)

    def open_path(self, p):
        busy = Busy(self.root, "Opening project")
        try:
            S = Session(p, progress=busy)
        except Exception as ex:
            busy.close(); traceback.print_exc(); messagebox.showerror("Open", f"Could not open {p}\n\n{type(ex).__name__}: {ex}"); return
        busy.close(); self.set_session(S); self.add_recent(p)

    def save(self):
        if not self.S:
            return False
        if not self.S.path:
            return self.save_as()
        if not self._guard_ok(self.S.save):
            return False
        self.add_recent(self.S.path); self.refresh_status(); return True

    def save_as(self):
        if not self.S:
            return False
        p = filedialog.asksaveasfilename(title="Save project as", defaultextension=".mistie.json",
                                         initialdir=os.path.dirname(self.S.default_save), initialfile=os.path.basename(self.S.path or self.S.default_save),
                                         filetypes=[("Mistie project", "*.mistie.json")])
        if not p:
            return False
        if not self._guard_ok(lambda: self.S.save(p)):
            return False
        self.add_recent(p); self.refresh_status(); return True

    def export_csv(self):
        if not self.S:
            return
        d = filedialog.askdirectory(title="Folder for ties.csv, corrections.csv, loops.csv", initialdir=self.S.P.out_dir)
        if d:
            self._guard(lambda: self.S.export_csv(d)); self.status.config(text=f"Exported tables to {d}")

    def write_segy(self):
        if not self.S:
            return
        d = filedialog.askdirectory(title="Folder for corrected SEG-Y copies", initialdir=self.S.P.out_dir)
        if not d:
            return
        if not messagebox.askokcancel("Write corrected SEG-Y", f"Write {len(self.S.names)} corrected copies (IEEE float, original headers) to\n{d}?"):
            return
        busy = Busy(self.root, "Writing SEG-Y"); busy("Applying corrections...")
        try:
            out = self.S.write_corrected_segy(d)
        except Exception as ex:
            busy.close(); messagebox.showerror("Write SEG-Y", str(ex)); return
        busy.close(); messagebox.showinfo("Write SEG-Y", "Wrote:\n" + "\n".join(os.path.basename(p) for p in out))

    def export_report(self):
        if not self.S:
            return
        p = filedialog.asksaveasfilename(title="Save report", defaultextension=".html", initialdir=self.S.P.out_dir,
                                         initialfile="mistie_report.html", filetypes=[("HTML", "*.html")])
        if p:
            self._guard(lambda: write_report(self, p)); self.status.config(text=f"Report written to {p}")

    def add_recent(self, p):
        try:
            r = json.load(open(RECENT))
        except Exception:
            r = []
        p = os.path.abspath(p); r = [p] + [x for x in r if x != p]
        try:
            json.dump(r[:8], open(RECENT, "w"))
        except OSError:
            pass
        self._recent_menu()

    def _recent_menu(self):
        self.recent_menu.delete(0, "end")
        try:
            r = json.load(open(RECENT))
        except Exception:
            r = []
        for p in r:
            self.recent_menu.add_command(label=p, command=lambda p=p: self.confirm_discard() and self.open_path(p))
        if not r:
            self.recent_menu.add_command(label="(none)", state="disabled")

    def on_close(self):
        if self.confirm_discard():
            self.root.destroy()

    # ======================================================== session plumbing
    def set_session(self, S):
        self.S, self.cur, self.map_lims, self.sort_col, self._map_drawn = S, 0, None, None, False
        self.solve_by.set(S.solve_by)
        self.colors = {L: LINE_COLORS[i % len(LINE_COLORS)] for i, L in enumerate(S.names)}
        self.refresh_all(rebuild_panels=True)

    def edit(self, fn):
        """Run an undoable session edit, then refresh."""
        if not self.S:
            return
        r = self._guard(fn)
        self.refresh_all(rebuild_panels=True)
        return r

    def refresh_all(self, rebuild_panels=False):
        if not self.S:
            return
        if rebuild_panels:
            self.load_panels()
        self.fill_ties(); self.fill_lines(); self.fill_loops(); self.fill_warnings(); self._ref_menu(); self._scen_menus()
        self.draw_map(); self.draw_tie(); self.draw_overlay(); self.refresh_status()

    def refresh_status(self):
        if not self.S:
            return
        s = self.S.summary()
        name = os.path.basename(self.S.path) if self.S.path else "(unsaved project)"
        self.root.title(f"Mistie Tool - {name}{' *' if self.S.dirty else ''}")
        self.status.config(text=f"  {name}{' (modified)' if self.S.dirty else ''}   |   {len(self.S.names)} lines, {s['n_ties']} ties, "
                                f"{s['n_used']} used   |   RMS mistie {s['before_ms']:.1f} -> {s['after_ms']:.1f} ms, "
                                f"{s['before_deg']:.0f} -> {s['after_deg']:.0f} deg   |   reference {self.S.P.ref}, solve per {self.S.solve_by}"
                                f"   |   undo {len(self.S.undo_stack)}")

    def selected_idxs(self):
        sel = [int(i) for i in self.tt.selection()]
        return sel or [self.cur]

    # ======================================================== tables
    def fill_ties(self):
        S, use = self.S, self.S.P.use_flags()
        rows = []
        for i, (t, u) in enumerate(zip(S.P.ties, use)):
            mis = u and S.is_misfit(t)
            rows.append((i, (f"{t['a']} x {t['b']}" + (" (end)" if t["near_end"] else ""), f"{t['window'][0]:.2f}-{t['window'][1]:.2f}{'*' if t['picked'] else ''}",
                             f"{t['shift']:+.1f}", f"{t['phase']:+.0f}", f"{t['gain']:.2f}", f"{t['corr']:.2f}", f"{t['half_diff']:.1f}", t["use"],
                             f"{t['res_shift']:+.1f}", f"{t['res_phase']:+.0f}", "misfit" if mis else ("fits" if u else "unused")),
                         ("misfit",) if mis else (() if u else ("unused",))))
        if self.sort_col:
            ci = self.tt["columns"].index(self.sort_col)

            def keyf(r):
                v = r[1][ci]
                try:
                    if self.sort_col == "window":
                        return float(str(v).split("-")[0])
                    f = float(v)
                    return abs(f) if self.sort_col in ("res_ms", "res_deg", "shift", "phase") else f
                except ValueError:
                    return str(v)
            if self.sort_col == "status":
                rk = {i: k for k, i in enumerate(S.ranking())}
                rows.sort(key=lambda r: rk[r[0]], reverse=self.sort_rev)
            else:
                rows.sort(key=keyf, reverse=self.sort_rev)
        keep = set(self.tt.selection())
        self.tt.delete(*self.tt.get_children())
        for i, vals, tags in rows:
            self.tt.insert("", "end", iid=str(i), values=vals, tags=tags)
        keep = [k for k in keep if self.tt.exists(k)] or [str(self.cur)]
        self._suppress = True
        self.tt.selection_set(keep); self.tt.see(str(self.cur)); self.tt.focus(str(self.cur))
        self.root.after_idle(lambda: setattr(self, "_suppress", False))

    def sort_ties(self, col):
        if self.sort_col == col:
            self.sort_rev = not self.sort_rev
        else:
            self.sort_col, self.sort_rev = col, col in ("res_ms", "res_deg", "half", "shift", "phase")
        self.fill_ties()

    def on_tie_select(self, e=None):
        if getattr(self, "_suppress", False) or not self.S:
            return
        foc = self.tt.focus()
        if foc and int(foc) != self.cur:
            self.select(int(foc), from_table=True)

    def fill_lines(self):
        S = self.S
        self.lt.delete(*self.lt.get_children())
        for L in S.names:
            c, st = S.P.sol[L], S.lines_state[L]
            if L == S.P.ref and not st["lock"]:
                status = "reference (held at zero)"
            elif st["lock"]:
                status = "locked at manual values"
            elif c["n_ties"] == 0:
                status = "NO USABLE TIES - undetermined"
            elif c["n_ties"] == 1:
                status = "one tie only - no independent check"
            else:
                status = "solved"
            self.lt.insert("", "end", iid=L, values=(L, S.survey_of[L], f"{c['shift']:+.2f}", f"{c['phase']:+.1f}", f"{c['gain']:.3f}", c["n_ties"], status))

    def fill_loops(self):
        self.lp.delete(*self.lp.get_children())
        if not self.S.P.loops:
            self.lp.insert("", "end", values=("No closed loops with the ties in use - the solution has no internal check", "", "", ""), tags=("bad",))
        for lp in self.S.P.loops:
            bad = abs(lp["shift"]) > 5 or abs(lp["phase"]) > 40
            self.lp.insert("", "end", values=(" > ".join(lp["lines"] + [lp["lines"][0]]), f"{lp['shift']:+.1f}", f"{lp['phase']:+.0f}",
                                             "DOES NOT CLOSE" if bad else "closes"), tags=("bad",) if bad else ())

    def fill_warnings(self):
        w = self.S.warnings()
        self.warn.config(state="normal"); self.warn.delete("1.0", "end")
        self.warn.insert("end", "\n".join(w) if w else "No warnings."); self.warn.config(state="disabled")
        self.nb.tab(2, text=f"Warnings ({len(w)})" if w else "Warnings")

    def _ref_menu(self):
        self.ref_menu.delete(0, "end"); self.ref_var = tk.StringVar(value=self.S.P.ref)
        for L in self.S.names:
            self.ref_menu.add_radiobutton(label=L, variable=self.ref_var, value=L, command=lambda L=L: self.edit(lambda: self.S.set_reference(L)))

    def _scen_menus(self):
        for menu, fn in [(self.scen_menu, self.load_scenario), (self.scen_del_menu, self.delete_scenario)]:
            menu.delete(0, "end")
            for n in self.S.scenarios:
                menu.add_command(label=n, command=lambda n=n, fn=fn: fn(n))
            if not self.S.scenarios:
                menu.add_command(label="(none saved)", state="disabled")

    # ======================================================== actions
    def select(self, k, from_table=False):
        self.cur = k % len(self.S.P.ties)
        self.load_panels()
        if not from_table:
            self._suppress = True
            self.tt.selection_set(str(self.cur)); self.tt.focus(str(self.cur)); self.tt.see(str(self.cur))
            self.root.after_idle(lambda: setattr(self, "_suppress", False))
        self.draw_map(); self.draw_tie(); self.draw_overlay()

    def step(self, d):
        order = [int(i) for i in self.tt.get_children()]
        if order:
            i = order.index(self.cur) if self.cur in order else 0
            self.select(order[(i + d) % len(order)])

    def next_worst(self):
        self.select(self.S.next_worst(self.cur))

    def set_view(self, v):
        self.view.set(v); self.draw_tie()

    def undo(self):
        if self.S and self.S.undo():
            self.refresh_all(rebuild_panels=True)
        else:
            self.status.config(text="Nothing to undo")

    def redo(self):
        if self.S and self.S.redo():
            self.refresh_all(rebuild_panels=True)
        else:
            self.status.config(text="Nothing to redo")

    def use_selected(self, mode):
        idx = self.selected_idxs(); self.edit(lambda: self.S.set_use(idx, mode))

    def reset_selected(self):
        idx = self.selected_idxs(); self.edit(lambda: self.S.reset_windows(idx))

    def copy_window(self):
        if not self.S:  # no project open
            return
        w = list(self.S.P.ties[self.cur]["window"]); idx = self.selected_idxs()
        self.edit(lambda: self.S.set_window(idx, w))

    def window_dialog(self, which):
        if not self.S:
            return
        d = WindowDialog(self.root, "Window for " + ("ALL ties" if which == "all" else f"{len(self.selected_idxs())} selected tie(s)"),
                         self.S.P.ties[self.cur]["window"])
        if d.result:
            idx = list(range(len(self.S.P.ties))) if which == "all" else self.selected_idxs()
            self.edit(lambda: self.S.set_window(idx, d.result))

    def reject_worst(self):
        if not self.S:  # no project open
            return
        i = self.edit(self.S.reject_worst)
        t = self.S.P.ties[i] if i is not None else None
        self.status.config(text=f"Excluded {t['a']} x {t['b']}" if t else "Worst tie not excluded: it is the only link for a line")

    def auto_reject(self):
        if not self.S:  # no project open
            return
        out = self.edit(self.S.auto_reject) or []
        messagebox.showinfo("Auto-reject", ("Excluded:\n" + "\n".join(f"{self.S.P.ties[i]['a']} x {self.S.P.ties[i]['b']}" for i in out))
                            if out else "No auto tie is far enough out of line to reject.")

    def all_auto(self):
        if not self.S:  # no project open
            return
        self.edit(self.S.all_auto)

    def set_solve_by(self):
        if not self.S:  # no project open
            return
        v = self.solve_by.get()
        if v == "survey" and len(set(self.S.survey_of.values())) == len(self.S.names):
            messagebox.showinfo("Surveys", "Every line is its own survey. Add \"survey\": \"NAME\" to the lines in the project's config "
                                           "(or right-click a line > Set survey) to group vintages.")
        self.edit(lambda: self.S.set_solve_by(v))

    def save_scenario(self):
        if not self.S:
            return
        d = tk.Toplevel(self.root); d.title("Save scenario"); d.transient(self.root)
        ttk.Label(d, text="Scenario name").pack(padx=12, pady=(12, 4)); e = ttk.Entry(d, width=30); e.pack(padx=12); e.focus_set()

        def ok():
            n = e.get().strip()
            if n:
                self.S.save_scenario(n); self._scen_menus(); self.refresh_status(); self.status.config(text=f"Saved scenario '{n}'")
            d.destroy()
        ttk.Button(d, text="Save", command=ok).pack(pady=10); d.bind("<Return>", lambda ev: ok()); d.grab_set()

    def load_scenario(self, n):
        self.edit(lambda: self.S.load_scenario(n)); self.status.config(text=f"Loaded scenario '{n}' (undo with Ctrl+Z)")

    def delete_scenario(self, n):
        if messagebox.askyesno("Delete scenario", f"Delete scenario '{n}'?"):
            self.S.delete_scenario(n); self._scen_menus(); self.refresh_status()

    def settings(self):
        if not self.S:
            return
        d = SettingsDialog(self.root, self.S.P.cfg)
        if not d.result:
            return
        remeasure = any(d.result[k] != self.S.P.cfg[k] for k in ("max_lag_ms", "tie_avg_traces"))
        self.S.P.cfg.update(d.result); self.S.cfg_raw.update(d.result); self.S.dirty = True
        if remeasure:
            self.S.P._mcache = {}
            for t in self.S.P.ties:
                self.S.P.measure(t)
        self.S.solve(); self.refresh_all(rebuild_panels=True)

    def tie_menu(self, e):
        row = self.tt.identify_row(e.y)
        if row and row not in self.tt.selection():
            self.tt.selection_set(row); self.tt.focus(row)
        m = tk.Menu(self.root, tearoff=0); n = len(self.selected_idxs())
        m.add_command(label=f"Use: auto ({n})", command=lambda: self.use_selected("auto"))
        m.add_command(label=f"Force on ({n})", command=lambda: self.use_selected("on"))
        m.add_command(label=f"Exclude ({n})", command=lambda: self.use_selected("off"))
        m.add_separator()
        m.add_command(label=f"Set window... ({n})", command=lambda: self.window_dialog("selected"))
        m.add_command(label="Copy current tie's window here", command=self.copy_window)
        m.add_command(label=f"Reset window ({n})", command=self.reset_selected)
        m.tk_popup(e.x_root, e.y_root)

    def line_menu(self, e):
        row = self.lt.identify_row(e.y)
        if not row:
            return
        self.lt.selection_set(row); L = row; st = self.S.lines_state[L]
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="Set as reference", command=lambda: self.edit(lambda: self.S.set_reference(L)))
        m.add_command(label="Lock at current solution", command=lambda: self.edit(lambda: self.S.lock_at_current(L)))
        m.add_command(label="Edit / lock manual correction...", command=self.edit_line)
        if st["lock"]:
            m.add_command(label="Unlock", command=lambda: self.edit(lambda: self.S.set_lock(L, False)))
        m.add_command(label="Set survey...", command=lambda: self.set_survey(L))
        m.add_separator()
        m.add_command(label="Show ties on this line", command=lambda: self.filter_line(L))
        m.tk_popup(e.x_root, e.y_root)

    def edit_line(self):
        sel = self.lt.selection()
        if not sel:
            return
        L = sel[0]; d = LockDialog(self.root, L, self.S.lines_state[L], self.S.P.sol[L])
        if d.result:
            lock, v = d.result; self.edit(lambda: self.S.set_lock(L, lock, v["shift"], v["phase"], v["gain"]))

    def set_survey(self, L):
        d = tk.Toplevel(self.root); d.title(f"Survey for {L}"); d.transient(self.root)
        ttk.Label(d, text="Survey / vintage name (lines with the same name can share one correction)").pack(padx=12, pady=(12, 4))
        e = ttk.Entry(d, width=30); e.insert(0, self.S.survey_of[L]); e.pack(padx=12); e.focus_set()

        def ok():
            v = e.get().strip() or L
            self.S.survey_of[L] = v
            for item in self.S.cfg_raw["lines"]:
                if item["name"] == L:
                    item["survey"] = v
            self.S.dirty = True; self.S.solve(); d.destroy(); self.refresh_all()
        ttk.Button(d, text="OK", command=ok).pack(pady=10); d.bind("<Return>", lambda ev: ok()); d.grab_set()

    def filter_line(self, L):
        ids = [str(i) for i, t in enumerate(self.S.P.ties) if L in (t["a"], t["b"])]
        self._suppress = True; self.tt.selection_set(ids)
        if ids:
            self.tt.see(ids[0])
        self.root.after_idle(lambda: setattr(self, "_suppress", False))
        self.status.config(text=f"Selected the {len(ids)} ties on {L}")

    # ======================================================== map
    def fit_map(self):
        self.map_lims = None; self._map_drawn = False; self.draw_map()

    def on_map_click(self, e):
        if not self.S or e.inaxes is None or e.xdata is None or self.map_tb.mode:
            return
        ax = e.inaxes; xr = np.ptp(ax.get_xlim()); yr = np.ptp(ax.get_ylim())
        d = [np.hypot((t["x"] - e.xdata) / xr, (t["y"] - e.ydata) / yr) for t in self.S.P.ties]
        k = int(np.argmin(d))
        if d[k] < 0.05:
            self.select(k)

    def draw_map(self):
        if not self.S:
            return
        S, use = self.S, self.S.P.use_flags()
        if getattr(self, "_map_drawn", False) and self.map_fig.axes:
            ax0 = self.map_fig.axes[0]; self.map_lims = (ax0.get_xlim(), ax0.get_ylim())
        self.map_fig.clear(); ax = self.map_fig.add_subplot(111)
        for L in S.names:
            g = S.P.geom[L]
            ax.plot(g[:, 1], g[:, 2], color=self.colors[L], lw=1.6, alpha=0.6 if MAP_MODES[self.map_mode.get()] else 1, zorder=1)
            ax.annotate(L, (g[0, 1], g[0, 2]), color=self.colors[L], fontsize=8, fontweight="bold", xytext=(2, 2), textcoords="offset points")
        mode, lab = MAP_MODES[self.map_mode.get()], self.map_labels.get()
        xs = np.array([t["x"] for t in S.P.ties]); ys = np.array([t["y"] for t in S.P.ties])
        if mode is None:
            cols = ["none" if not u else ("#c92a2a" if S.is_misfit(t) else "#2b8a3e") for t, u in zip(S.P.ties, use)]
            ax.scatter(xs, ys, s=46, c=cols, edgecolors="0.3", linewidths=0.8, zorder=3)
        else:
            lim = 180.0 if "phase" in mode else 20.0
            vals = np.array([t[mode] for t in S.P.ties], float)
            sc = ax.scatter(xs, ys, c=np.abs(vals), cmap="YlOrRd", vmin=0, vmax=lim, s=30 + 70 * np.clip(np.abs(vals) / lim, 0, 1),
                            edgecolors=["0.25" if u else "0.6" for u in use], linewidths=[0.9 if u else 0.8 for u in use],
                            linestyles=["-" if u else "--" for u in use], zorder=3)
            cb = self.map_fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.set_label(("|" + mode.replace("_", " ") + "|") + (" deg" if "phase" in mode else " ms"), fontsize=8)
            cb.ax.tick_params(labelsize=7)
            for i, (t, v, u) in enumerate(zip(S.P.ties, vals, use)):
                show = lab == "all" or (lab == "selected only" and i == self.cur) or \
                       (lab == "misfits + selected" and (i == self.cur or (u and S.is_misfit(t)) or abs(v) > lim * 0.5))
                if show:
                    ax.annotate(f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}", (t["x"], t["y"]), xytext=(5, 3), textcoords="offset points",
                                fontsize=7.5, color="black" if u else "0.5", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        t = S.P.ties[self.cur]
        ax.scatter([t["x"]], [t["y"]], s=170, facecolors="none", edgecolors="black", linewidths=2, zorder=4)
        ax.set_aspect("equal", adjustable="box" if self.map_lims else "datalim"); ax.tick_params(labelsize=7)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(self.map_mode.get() + ("" if mode else "  (green fits, red misfit, hollow unused)"), fontsize=9)
        if self.map_lims:
            ax.set_xlim(self.map_lims[0]); ax.set_ylim(self.map_lims[1])
        self.map_fig.tight_layout(); self.map_canvas.draw_idle(); self._map_drawn = True

    # ======================================================== tie panel
    def load_panels(self):
        t = self.S.P.ties[self.cur]
        self.pA, self.pB = self.S.P.panels(t)
        self.clip0 = float(np.percentile(np.abs(np.vstack([self.pA, self.pB])), 98)) or 1.0

    def corrected(self):
        t, P, dt, v = self.S.P.ties[self.cur], self.S.P, self.S.P.dt, self.view.get()
        A, B = self.pA, self.pB
        if v == "tie":
            B = apply_correction(B, dt, t["shift"], t["phase"], t["gain"])
        elif v == "net":
            a, b = P.sol[t["a"]], P.sol[t["b"]]
            A = apply_correction(A, dt, a["shift"], a["phase"], a["gain"]); B = apply_correction(B, dt, b["shift"], b["phase"], b["gain"])
        return np.atleast_2d(A), np.atleast_2d(B)

    def draw_tie(self):
        if not self.S:
            return
        S, t, dt = self.S, self.S.P.ties[self.cur], self.S.P.dt
        A, B = self.corrected(); ns = A.shape[1]
        auto_t = max(3.0, 1.3 * max(tt["window"][1] for tt in S.P.ties))
        tmax = min(S.P.cfg.get("display_tmax") or auto_t, ns * dt); self.tmax = tmax
        clip = self.clip0 / max(self.gain.get(), 0.05)
        self.tie_fig.clear(); ax = self.tie_ax = self.tie_fig.add_subplot(111)
        img = np.vstack([A, B])[:, : int(tmax / dt)]
        ax.imshow(img.T, aspect="auto", cmap="gray_r", vmin=-clip, vmax=clip, extent=[-len(A) + 0.5, len(B) + 0.5, tmax, 0], interpolation="bilinear")
        ax.axvline(0.5, color=TIE_C, ls="--", lw=1.4)
        w = t["window"]
        ax.axhspan(0, w[0], color="k", alpha=0.3, zorder=2); ax.axhspan(w[1], tmax, color="k", alpha=0.3, zorder=2)
        self.top_line = ax.axhline(w[0], color=TOP_C, lw=2.5, zorder=3); self.base_line = ax.axhline(w[1], color=BASE_C, lw=2.5, zorder=3)
        kw = dict(transform=ax.get_yaxis_transform(), ha="right", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
        self.top_txt = ax.text(0.99, w[0], f"Top {w[0]:.2f} s", bbox=dict(fc=TOP_C, ec="none"), **kw)
        self.base_txt = ax.text(0.99, w[1], f"Base {w[1]:.2f} s", bbox=dict(fc=BASE_C, ec="none"), **kw)
        ax.set_xlabel(f"<-- {t['a']} approaching (to CDP {t['ca']:.0f})        |        {t['b']} leaving (from CDP {t['cb']:.0f}) -->")
        ax.set_ylabel("TWT (s)")
        self.tie_fig.tight_layout(); self.tie_canvas.draw_idle()
        view = {"raw": "raw", "tie": "tie-corrected", "net": "network-corrected"}[self.view.get()]
        self.tie_title.config(text=f"Tie {self.cur + 1} of {len(S.P.ties)}:  {t['a']} CDP {t['ca']:.0f}  x  {t['b']} CDP {t['cb']:.0f}"
                                   f"{'   (near a line end)' if t['near_end'] else ''}      view: {view}")

    def set_pick(self, which, tt):
        t = self.S.P.ties[self.cur]; w = list(t["window"]); tt = float(np.clip(tt, 0.05, self.tmax - 0.05))
        if which == 0:
            w[0] = min(tt, w[1] - 0.2)
        else:
            w[1] = max(tt, w[0] + 0.2)
        return w

    def show_pick(self, w):
        self.top_line.set_ydata([w[0], w[0]]); self.base_line.set_ydata([w[1], w[1]])
        self.top_txt.set_y(w[0]); self.top_txt.set_text(f"Top {w[0]:.2f} s"); self.base_txt.set_y(w[1]); self.base_txt.set_text(f"Base {w[1]:.2f} s")
        self.tie_canvas.draw_idle()

    def on_press(self, e):
        if not self.S or e.inaxes is not getattr(self, "tie_ax", None) or e.ydata is None or e.button != 1:
            return
        w = self.S.P.ties[self.cur]["window"]
        self.drag = 0 if abs(e.ydata - w[0]) <= abs(e.ydata - w[1]) else 1
        self.drag_w = self.set_pick(self.drag, e.ydata); self.show_pick(self.drag_w)

    def on_move(self, e):
        if e.inaxes is getattr(self, "tie_ax", None):
            self.mouse_t = e.ydata
        if self.drag is not None and e.ydata is not None and e.inaxes is getattr(self, "tie_ax", None):
            self.drag_w = self.set_pick(self.drag, e.ydata); self.show_pick(self.drag_w)

    def on_release(self, e):
        if self.drag is None:
            return
        self.drag = None; w = self.drag_w; k = self.cur
        self.edit(lambda: self.S.set_window([k], w))

    def key_pick(self, which):
        if self.mouse_t is None:
            self.status.config(text="Move the mouse over the tie panel first"); return
        w = self.set_pick(which, self.mouse_t); k = self.cur
        self.edit(lambda: self.S.set_window([k], w))

    # ======================================================== overlay
    def draw_overlay(self):
        if not self.S:
            return
        S, t, dt = self.S, self.S.P.ties[self.cur], self.S.P.dt
        a, b, _, _ = S.P.tie_traces(t); w = t["window"]
        bc = apply_correction(b, dt, t["shift"], t["phase"], t["gain"])
        i1 = min(int(w[1] / dt), len(a), len(b)); i0 = min(int(w[0] / dt), max(0, i1 - 2)); tt = np.arange(i0, i1) * dt
        m = max(np.abs(a[i0:i1]).max(), np.abs(b[i0:i1]).max(), 1e-12)
        self.ov_fig.clear(); ax = self.ov_fig.add_subplot(111)
        ax.plot(a[i0:i1] / m, tt, "k", lw=0.9); ax.plot(b[i0:i1] / m, tt, color="#d9480f", lw=0.9, alpha=0.85)
        ax.plot(a[i0:i1] / m + 2.4, tt, "k", lw=0.9); ax.plot(bc[i0:i1] / m + 2.4, tt, color="#1c64d6", lw=0.9, alpha=0.85)
        ax.set_ylim(w[1], w[0]); ax.set_xlim(-1.3, 3.7); ax.set_xticks([0, 2.4]); ax.set_xticklabels(["before", "after"], fontsize=8)
        ax.tick_params(labelsize=7); ax.set_title(f"{t['a']} black,  {t['b']} red / blue", fontsize=9)
        self.ov_fig.tight_layout(); self.ov_canvas.draw_idle()
        c = S.P.cfg; u = S.P.use_flags()[self.cur]
        ok = lambda x: "ok" if x else "CHECK"
        txt = (f"Shift        {t['shift']:+7.1f} ms\nPhase        {t['phase']:+7.0f} deg\nGain         {t['gain']:7.2f}\n"
               f"Correlation  {t['corr']:7.2f}  {ok(t['corr'] >= c['min_corr'])}\nHalves diff  {t['half_diff']:7.1f} ms {ok(t['half_diff'] <= c['max_half_diff_ms'])}\n"
               f"Use          {t['use']:>7s}  ({'used' if u else 'not used'})\n"
               f"Residual     {t['res_shift']:+7.1f} ms {t['res_phase']:+5.0f} deg\nWindow       {w[0]:.2f}-{w[1]:.2f} s {'(picked)' if t['picked'] else '(default)'}")
        self.meas.config(state="normal"); self.meas.delete("1.0", "end"); self.meas.insert("end", txt); self.meas.config(state="disabled")


# ============================================================ report
def write_report(app, path):
    import base64, io
    S = app.S; s = S.summary(); esc = html.escape

    def png(fig):
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    mode_before = app.map_mode.get()
    app.map_mode.set("Mistie: time shift (ms)"); app.draw_map(); img1 = png(app.map_fig)
    app.map_mode.set("Residual: time (ms)"); app.draw_map(); img2 = png(app.map_fig)
    app.map_mode.set(mode_before); app.draw_map()
    use = S.P.use_flags()
    rows = "".join(f"<tr><td>{esc(L)}</td><td>{esc(S.survey_of[L])}</td><td>{S.P.sol[L]['shift']:+.2f}</td><td>{S.P.sol[L]['phase']:+.1f}</td>"
                   f"<td>{S.P.sol[L]['gain']:.3f}</td><td>{S.P.sol[L]['n_ties']}</td><td>{'reference' if L == S.P.ref else 'locked' if S.lines_state[L]['lock'] else ''}</td></tr>"
                   for L in S.names)
    trows = "".join(f"<tr class='{'bad' if u and S.is_misfit(t) else '' if u else 'off'}'><td>{esc(t['a'])} x {esc(t['b'])}</td><td>{t['ca']:.0f} / {t['cb']:.0f}</td>"
                    f"<td>{t['window'][0]:.2f}-{t['window'][1]:.2f}</td><td>{t['shift']:+.1f}</td><td>{t['phase']:+.0f}</td><td>{t['corr']:.2f}</td>"
                    f"<td>{t['use']}</td><td>{t['res_shift']:+.1f}</td><td>{t['res_phase']:+.0f}</td></tr>" for t, u in zip(S.P.ties, use))
    lrows = "".join(f"<tr><td>{esc(' > '.join(lp['lines'] + [lp['lines'][0]]))}</td><td>{lp['shift']:+.1f}</td><td>{lp['phase']:+.0f}</td></tr>" for lp in S.P.loops) \
        or "<tr><td colspan=3>No closed loops</td></tr>"
    warn = "".join(f"<li>{esc(w)}</li>" for w in S.warnings()) or "<li>None</li>"
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Mistie report</title><style>
body{{font:14px/1.45 Segoe UI,Arial,sans-serif;margin:28px;color:#14181b;max-width:1100px}}h1{{font-size:26px;margin:0 0 4px}}
table{{border-collapse:collapse;margin:8px 0 18px}}td,th{{border-bottom:1px solid #ddd;padding:4px 10px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
tr.bad td{{color:#c92a2a}}tr.off td{{color:#888}}.imgs{{display:flex;gap:16px;flex-wrap:wrap}}.imgs img{{max-width:520px;width:100%}}</style></head><body>
<h1>Mistie report</h1><p>{esc(os.path.basename(S.path or 'unsaved project'))} - {len(S.names)} lines, {s['n_ties']} ties, {s['n_used']} used.
Reference line {esc(S.P.ref)}, one correction per {S.solve_by}.<br>RMS mistie {s['before_ms']:.1f} &rarr; {s['after_ms']:.1f} ms,
{s['before_deg']:.0f} &rarr; {s['after_deg']:.0f} deg after network correction.</p>
<div class="imgs"><img src="{img1}" alt="measured misties"><img src="{img2}" alt="residual misties"></div>
<h2>Corrections per line</h2><table><tr><th>Line</th><th>Survey</th><th>Shift ms</th><th>Phase deg</th><th>Gain</th><th>Ties used</th><th></th></tr>{rows}</table>
<h2>Warnings</h2><ul>{warn}</ul><h2>Loop closures</h2><table><tr><th>Loop</th><th>ms</th><th>deg</th></tr>{lrows}</table>
<h2>Ties</h2><table><tr><th>Tie</th><th>CDPs</th><th>Window s</th><th>Shift ms</th><th>Phase</th><th>r</th><th>Use</th><th>Resid ms</th><th>Resid deg</th></tr>{trows}</table>
<p style="color:#666">Convention: a line's correction is applied as delay by shift (positive = later), constant phase rotation, then gain.
Generated by Mistie Tool {__version__}.</p></body></html>"""
    open(path, "w", encoding="utf-8").write(doc)


def main(path=None):
    root = tk.Tk()
    MistieApp(root, path)
    root.mainloop()
