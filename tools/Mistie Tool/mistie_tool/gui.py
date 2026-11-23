"""Interactive tie picker (matplotlib). Runs fully offline."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from .measure import apply_correction
from . import network as net

TOP_C, BASE_C = "#16b3c9", "#d9468f"
HELP = ("Drag cyan/magenta lines or click to move the nearest  |  t / b = top / base at cursor  |  "
        "n / p = next / previous tie  |  u = cycle use (auto/on/off)  |  r = reset window  |  "
        "w = reject worst tie  |  1 2 3 = raw / tie / network view  |  s = save")


class App:
    def __init__(self, project):
        self.P, self.k, self.mode, self.drag, self.mouse_t = project, 0, "raw", None, None
        P = project
        P.solve()
        self.fig = plt.figure(figsize=(17, 10))
        try:
            self.fig.canvas.manager.set_window_title("Mistie tool")
        except Exception:
            pass
        gs = self.fig.add_gridspec(3, 3, width_ratios=[1.0, 2.2, 0.9], height_ratios=[1, 1, 0.72],
                                   left=0.04, right=0.985, top=0.9, bottom=0.05, wspace=0.18, hspace=0.32)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_list = self.fig.add_subplot(gs[1, 0]); self.ax_list.axis("off")
        self.ax_seis = self.fig.add_subplot(gs[0:2, 1])
        self.ax_wig = self.fig.add_subplot(gs[0:2, 2])
        self.ax_net = self.fig.add_subplot(gs[2, :2]); self.ax_net.axis("off")
        self.ax_res = self.fig.add_subplot(gs[2, 2]); self.ax_res.axis("off")
        # widgets
        def btn(x, label, cb, w=0.055):
            b = Button(self.fig.add_axes([x, 0.935, w, 0.035]), label); b.on_clicked(cb); return b
        self.buttons = [btn(0.04, "< Prev", lambda e: self.select(self.k - 1)),
                        btn(0.10, "Next >", lambda e: self.select(self.k + 1)),
                        btn(0.17, "Use: cycle", lambda e: self.cycle_use(), 0.065),
                        btn(0.24, "Reset window", lambda e: self.reset(), 0.07),
                        btn(0.32, "Reject worst", lambda e: self.reject_worst(), 0.07),
                        btn(0.40, "All ties auto", lambda e: self.all_auto(), 0.07),
                        btn(0.48, "Save", lambda e: self.save(), 0.05)]
        self.radio = RadioButtons(self.fig.add_axes([0.56, 0.925, 0.2, 0.055], frameon=False),
                                  ("raw", "tie-corrected", "network-corrected"))
        self.radio.on_clicked(self.set_mode)
        self.fig.text(0.04, 0.012, HELP, fontsize=8.5, color="0.35")
        self.status = self.fig.text(0.985, 0.012, "", fontsize=9, ha="right", color="#1d6f42")
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: self.save(quiet=True))
        self.select(0)

    # ---------------- state changes
    @property
    def t(self):
        return self.P.ties[self.k]

    def select(self, k):
        self.k = k % len(self.P.ties)
        self.pA, self.pB = self.P.panels(self.t)
        self.clip = np.percentile(np.abs(np.vstack([self.pA, self.pB])), 98) or 1.0
        self.redraw()

    def changed(self, remeasure=True):
        if remeasure:
            self.P.measure(self.t)
        self.P.solve(); self.P.save(); self.redraw()

    def set_mode(self, label):
        self.mode = {"raw": "raw", "tie-corrected": "tie", "network-corrected": "net"}[label]; self.redraw()

    def cycle_use(self):
        self.t["use"] = {"auto": "on", "on": "off", "off": "auto"}[self.t["use"]]; self.changed(False)

    def reset(self):
        self.t["window"] = list(self.P.cfg["default_window"]); self.t["picked"] = False; self.changed()

    def all_auto(self):
        for t in self.P.ties:
            t["use"] = "auto"
        self.changed(False)

    def reject_worst(self):
        use = self.P.use_flags()
        cand = [(abs(t["res_shift"]) / self.P.cfg["reject_tol_ms"] + abs(t["res_phase"]) / self.P.cfg["reject_tol_deg"], i)
                for i, (t, u) in enumerate(zip(self.P.ties, use)) if u]
        if not cand:
            return
        _, i = max(cand)
        trial = list(use); trial[i] = False
        if not net.connected(self.P.names, self.P.ties, trial):
            self.status.set_text("Worst tie not removed: it is the only link for a line"); self.fig.canvas.draw_idle(); return
        self.P.ties[i]["use"] = "off"
        self.status.set_text(f"Excluded {self.P.ties[i]['a']} x {self.P.ties[i]['b']}")
        self.changed(False)

    def save(self, quiet=False):
        self.P.solve(); d = self.P.save()
        if not quiet:
            self.status.set_text(f"Saved picks, ties.csv, corrections.csv, loops.csv to {d}"); self.fig.canvas.draw_idle()

    def set_pick(self, which, tt):
        w = self.t["window"]; tmax = self.tmax - 0.05
        tt = float(np.clip(tt, 0.05, tmax))
        if which == 0:
            w[0] = min(tt, w[1] - 0.2)
        else:
            w[1] = max(tt, w[0] + 0.2)
        self.t["picked"] = True

    # ---------------- events
    def on_press(self, e):
        if e.inaxes is self.ax_seis and e.ydata is not None and e.button == 1:
            w = self.t["window"]
            self.drag = 0 if abs(e.ydata - w[0]) <= abs(e.ydata - w[1]) else 1
            self.set_pick(self.drag, e.ydata); self.update_picks()
        elif e.inaxes is self.ax_map and e.xdata is not None:
            d = [np.hypot(t["x"] - e.xdata, t["y"] - e.ydata) for t in self.P.ties]
            self.select(int(np.argmin(d)))

    def on_move(self, e):
        if e.inaxes is self.ax_seis:
            self.mouse_t = e.ydata
        if self.drag is not None and e.inaxes is self.ax_seis and e.ydata is not None:
            self.set_pick(self.drag, e.ydata); self.update_picks()

    def on_release(self, e):
        if self.drag is not None:
            self.drag = None; self.changed()

    def on_key(self, e):
        k = (e.key or "").lower()
        if k in ("t", "b") and self.mouse_t is not None:
            self.set_pick(0 if k == "t" else 1, self.mouse_t); self.changed()
        elif k in ("n", "right"):
            self.select(self.k + 1)
        elif k in ("p", "left"):
            self.select(self.k - 1)
        elif k == "u":
            self.cycle_use()
        elif k == "r":
            self.reset()
        elif k == "w":
            self.reject_worst()
        elif k == "s":
            self.save()
        elif k in ("1", "2", "3"):
            self.radio.set_active(int(k) - 1)

    # ---------------- drawing
    def update_picks(self):
        w = self.t["window"]
        self.top_line.set_ydata([w[0], w[0]]); self.base_line.set_ydata([w[1], w[1]])
        self.top_txt.set_y(w[0]); self.top_txt.set_text(f"Top {w[0]:.2f} s")
        self.base_txt.set_y(w[1]); self.base_txt.set_text(f"Base {w[1]:.2f} s")
        self.fig.canvas.draw_idle()

    def corrected_panels(self):
        t, P, dt = self.t, self.P, self.P.dt
        A, B = self.pA, self.pB
        if self.mode == "tie":
            B = apply_correction(B, dt, t["shift"], t["phase"], t["gain"])
        elif self.mode == "net":
            a, b = P.sol[t["a"]], P.sol[t["b"]]
            A = apply_correction(A, dt, a["shift"], a["phase"], a["gain"])
            B = apply_correction(B, dt, b["shift"], b["phase"], b["gain"])
        return np.atleast_2d(A), np.atleast_2d(B)

    def redraw(self):
        P, t, dt = self.P, self.t, self.P.dt
        use = P.use_flags()
        tmax_cfg = P.cfg["display_tmax"]
        ns = self.pA.shape[1]
        self.tmax = min(tmax_cfg or ns * dt, ns * dt)
        # --- map
        ax = self.ax_map; ax.clear()
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, n in enumerate(P.names):
            g = P.geom[n]
            ax.plot(g[:, 1], g[:, 2], color=cols[i % len(cols)], lw=1.8)
            ax.annotate(n, (g[0, 1], g[0, 2]), color=cols[i % len(cols)], fontsize=8, fontweight="bold")
        for i, (tt, u) in enumerate(zip(P.ties, use)):
            bad = u and (abs(tt["res_shift"]) > P.cfg["reject_tol_ms"] or abs(tt["res_phase"]) > P.cfg["reject_tol_deg"])
            fc = "none" if not u else ("#d62728" if bad else "#2ca02c")
            ax.plot(tt["x"], tt["y"], "o", ms=11 if i == self.k else 7, mfc=fc,
                    mec="k" if i == self.k else "0.4", mew=2 if i == self.k else 1)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Map (click a tie)  green fits / red misfit / hollow unused", fontsize=9)
        # --- tie list
        ax = self.ax_list; ax.clear(); ax.axis("off")
        rows = ["   tie                window     shift  phase    r  use"]
        for i, (tt, u) in enumerate(zip(P.ties, use)):
            rows.append(f"{'>' if i == self.k else ' '} {tt['a'] + ' x ' + tt['b']:15s} {tt['window'][0]:4.2f}-{tt['window'][1]:4.2f}"
                        f"{'*' if tt['picked'] else ' '} {tt['shift']:+6.1f} {tt['phase']:+5.0f}  {tt['corr']:.2f}  "
                        f"{tt['use']}{'' if u else ' (x)'}")
        ax.text(0, 1, "\n".join(rows), family="monospace", fontsize=8, va="top", transform=ax.transAxes)
        ax.set_title("Ties  (* = picked window, (x) = not used)", fontsize=9, loc="left")
        # --- seismic tie panel
        ax = self.ax_seis; ax.clear()
        A, B = self.corrected_panels()
        img = np.vstack([A, B])[:, : int(self.tmax / dt)]
        ax.imshow(img.T, aspect="auto", cmap="gray_r", vmin=-self.clip, vmax=self.clip,
                  extent=[-len(A) + 0.5, len(B) + 0.5, self.tmax, 0], interpolation="bilinear")
        ax.axvline(0.5, color="#ffd43b", ls="--", lw=1.4)
        w = t["window"]
        ax.axhspan(0, w[0], color="k", alpha=0.35, zorder=3); ax.axhspan(w[1], self.tmax, color="k", alpha=0.35, zorder=3)
        self.top_line = ax.axhline(w[0], color=TOP_C, lw=2.5)
        self.base_line = ax.axhline(w[1], color=BASE_C, lw=2.5)
        kw = dict(transform=ax.get_yaxis_transform(), ha="right", va="center", color="white", fontsize=9, fontweight="bold")
        self.top_txt = ax.text(0.99, w[0], f"Top {w[0]:.2f} s", bbox=dict(fc=TOP_C, ec="none"), **kw)
        self.base_txt = ax.text(0.99, w[1], f"Base {w[1]:.2f} s", bbox=dict(fc=BASE_C, ec="none"), **kw)
        ax.set_xlabel(f"<-- {t['a']} approaching (to CDP {t['ca']:.0f})      |      {t['b']} leaving (from CDP {t['cb']:.0f}) -->")
        ax.set_ylabel("TWT (s)")
        ax.set_title(f"Tie {self.k + 1}/{len(P.ties)}:  {t['a']} CDP {t['ca']:.0f}  x  {t['b']} CDP {t['cb']:.0f}"
                     f"{'   (near line end)' if t['near_end'] else ''}     view: {self.mode}", fontsize=11)
        # --- wiggle overlay
        ax = self.ax_wig; ax.clear()
        a, b, _, _ = P.tie_traces(t)
        bc = apply_correction(b, dt, t["shift"], t["phase"], t["gain"])
        i1 = min(int(w[1] / dt), len(a), len(b)); i0 = min(int(w[0] / dt), max(0, i1 - 2)); tt_ = np.arange(i0, i1) * dt
        m = max(np.abs(a[i0:i1]).max(), np.abs(b[i0:i1]).max(), 1e-12)
        ax.plot(a[i0:i1] / m, tt_, "k", lw=0.9); ax.plot(b[i0:i1] / m, tt_, color="#d9480f", lw=0.9, alpha=0.85)
        ax.plot(a[i0:i1] / m + 2.4, tt_, "k", lw=0.9); ax.plot(bc[i0:i1] / m + 2.4, tt_, color="#1c64d6", lw=0.9, alpha=0.85)
        ax.set_ylim(w[1], w[0]); ax.set_xlim(-1.3, 3.7); ax.set_xticks([0, 2.4]); ax.set_xticklabels(["before", "after tie corr."])
        ok = lambda c: "OK" if c else "CHECK"
        ax.set_title(f"{t['a']} black, {t['b']} red/blue\nshift {t['shift']:+.1f} ms   phase {t['phase']:+.0f} deg   gain {t['gain']:.2f}\n"
                     f"corr {t['corr']:.2f} [{ok(t['corr'] >= P.cfg['min_corr'])}]   halves differ {t['half_diff']:.1f} ms "
                     f"[{ok(t['half_diff'] <= P.cfg['max_half_diff_ms'])}]   use: {t['use']}", fontsize=9)
        # --- network
        s = P.summary()
        rows = [f"NETWORK  reference {P.ref}   RMS mistie {s['before_ms']:.1f} -> {s['after_ms']:.1f} ms,  "
                f"{s['before_deg']:.0f} -> {s['after_deg']:.0f} deg   ({s['n_used']}/{s['n_ties']} ties used)", "",
                f"{'line':7s} {'shift ms':>9s} {'phase':>7s} {'gain':>6s}  ties"]
        for n in P.names:
            c = P.sol[n]
            note = "" if n == P.ref else (" unconstrained!" if c["n_ties"] == 0 else " (no check)" if c["n_ties"] == 1 else "")
            rows.append(f"{n:7s} {c['shift']:+9.1f} {c['phase']:+7.0f} {c['gain']:6.2f}  {c['n_ties']}{' ref' if n == P.ref else ''}{note}")
        rows += ["", "LOOP CLOSURE (raw misties summed around each loop)"]
        if not P.loops:
            rows.append("  no closed loops with the ties in use - the solution has no internal check")
        for lp in P.loops:
            bad = abs(lp["shift"]) > 5 or abs(lp["phase"]) > 40
            rows.append(f"  {' > '.join(lp['lines'] + [lp['lines'][0]]):40s} {lp['shift']:+6.1f} ms {lp['phase']:+5.0f} deg  "
                        f"{'DOES NOT CLOSE' if bad else 'closes'}")
        self.ax_net.clear(); self.ax_net.axis("off")
        self.ax_net.text(0, 1, "\n".join(rows), family="monospace", fontsize=8.5, va="top", transform=self.ax_net.transAxes)
        rows = ["RESIDUALS after network correction", f"{'tie':15s} {'ms':>6s} {'deg':>5s}"]
        for tt, u in zip(P.ties, use):
            flag = "" if not u else ("  <-- misfit" if abs(tt["res_shift"]) > P.cfg["reject_tol_ms"] or abs(tt["res_phase"]) > P.cfg["reject_tol_deg"] else "")
            rows.append(f"{tt['a'] + ' x ' + tt['b']:15s} {tt['res_shift']:+6.1f} {tt['res_phase']:+5.0f}{flag if u else '  (unused)'}")
        self.ax_res.clear(); self.ax_res.axis("off")
        self.ax_res.text(0, 1, "\n".join(rows), family="monospace", fontsize=8, va="top", transform=self.ax_res.transAxes)
        self.fig.canvas.draw_idle()


def run(project):
    app = App(project)
    plt.show()
    return app
