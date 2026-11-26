"""F3 worked example for the GeoHarmonic tutorial (PLAN 43h/i).

One continuous interpretation of the F3 Netherlands project, chapter by chapter in the order Chris works. Every
chapter RUNS the real workflow (the same code the windows call) on a small window of the data, takes its numbers
from that run and its pictures from the program (the tools' own exports; on the PC with real Qt also the window
itself), so the tutorial cannot drift from the program.

    project_path:  Chris's 'F3 Netherlands.seisproj.json' (Documents\\F3 Netherlands) - or, where it does not exist
                   (build server, a PC without F3), the synthetic F3-like project from seismic_tool.synth_f3, built by
                   tools/setup_f3.py exactly like the real one (same names, rotated F3 bin grid, 4 wells, Fault A).

Each chapter: dict(id, title, purpose, steps, look, run) - run(ctx) fills ctx.n (numbers quoted in the text with
{name}) and adds pictures with ctx.export(...) / ctx.window(...).  A chapter that fails is captioned with its error
and the others still run; build_tutorial.py writes a progress line per chapter and a log file."""
import os
import time
import traceback

import numpy as np

IL_FOCUS = 425  # the inline Chris uses on F3


def default_f3_path():
    up = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(up, "Documents", "F3 Netherlands", "F3 Netherlands.seisproj.json")


def working_copy(real_path, folder):
    """A copy of the seismic project JSON + its horizons in `folder` that points at the real volume stores and well
    project: the tutorial never writes into Chris's F3 project (Live grids, maps, temporary horizons go here)."""
    from seismic_tool.project import SeisProject
    p = SeisProject.open(real_path)
    os.makedirs(folder, exist_ok=True)
    for v in p.volumes:  # absolute before the move
        v["store"] = os.path.abspath(v["store"])
    p.path = os.path.join(folder, os.path.basename(real_path))
    p.horizons.dirty = True
    p.save()
    return p.path


# ============================================================================================ context
class F3Ctx:
    def __init__(self, project_path, kind, img_dir, mode, log=print, mw_factory=None, grab=None):
        self.path, self.kind, self.img_dir, self.mode, self.log = project_path, kind, img_dir, mode, log
        self.n = {}
        self.shots = []  # per chapter, reset by run_chapter
        self._mw = None
        self._p = None
        self.mw_factory = mw_factory
        self.grab = grab  # callable(win, path) -> marks (real Qt only)

    # ---- data
    def project(self):
        if self._p is None:
            from seismic_tool.project import SeisProject
            self._p = SeisProject.open(self.path)
        return self._p

    def mw(self):
        if self._mw is None:
            from seismic_tool.gui.main_window import MainWindow
            self._mw = MainWindow(self.path)
            self._p = self._mw.project
        return self._mw

    def vol(self):
        return self.project().volume(self.vol_name())

    def vol_name(self):
        p = self.project()
        names = [v["name"] for v in p.volumes]
        return "F3 original" if "F3 original" in names else names[0]

    def il_focus(self):
        v = self.vol()
        il = min(max(IL_FOCUS, int(v.inlines[0])), int(v.inlines[-1]))
        return int(v.inlines[int(np.argmin(np.abs(v.inlines - il)))])

    def horizon_names(self):
        key = f"3D:{self.vol_name()}"
        return [h.name for h in self.project().horizons.items if h.data.get(key) is not None]

    def hz(self, preferred):
        names = self.horizon_names()
        for p_ in preferred:
            for n in names:
                if n.lower() == p_.lower():
                    return n
        return names[len(names) // 2] if names else None

    def where(self):
        if self.kind == "f3":
            return "your F3 Netherlands project"
        return "the synthetic F3-like test project (same names and survey frame as F3; your F3 project was not found)"

    # ---- pictures
    def export(self, sid, title, fn, marks=()):
        """fn(path) draws the picture (matplotlib / the tool's own export). marks = [(fx, fy, label)]."""
        path = os.path.join(self.img_dir, f"f3_{sid}.png")
        res = fn(path)
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], tuple):
            marks = res
        self.shots.append(dict(id=f"f3_{sid}", title=title, file=path, marks=list(marks),
                               caption=f"{title} - {self.where()}, drawn by the program's own export."))
        return path

    def window(self, sid, title, win):
        if self.mode != "real" or self.grab is None or win is None:
            return None
        path = os.path.join(self.img_dir, f"f3_{sid}.png")
        try:
            marks = self.grab(win, path)
        except Exception as e:  # noqa: BLE001
            self.log(f"    window {sid} not captured: {e}")
            return None
        self.shots.append(dict(id=f"f3_{sid}", title=title, file=path, marks=marks,
                               caption=f"{title} - screenshot of the real window on {self.where()}."))
        return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _section_png(path, vol, il, horizons, project, title, faults=True, data=None, cmap="seismic"):
    plt = _plt()
    sec = np.asarray(vol.inline(il), float) if data is None else data
    t = np.asarray(vol.times, float)
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=100)
    c = np.nanpercentile(np.abs(sec), 98) or 1.0
    ax.imshow(sec.T, aspect="auto", cmap=cmap, vmin=-c, vmax=c, extent=[vol.xlines[0], vol.xlines[-1], t[-1], t[0]])
    i = vol.il_index(il)
    cols = ["#ffd400", "#00e0ff", "#7CFC00", "#ff66cc", "#ffffff", "#ff9900", "#b0a0ff"]
    for k, name in enumerate(horizons):
        h = project.horizons.get(name)
        z = h.data.get(f"3D:{vol_name_of(project, vol)}") if h is not None else None
        if z is not None:
            ax.plot(vol.xlines, np.asarray(z, float)[i], color=cols[k % len(cols)], lw=1.6, label=name)
    if faults:
        for f in project.faults.items:
            for xs_, ts_ in fault_cuts(f, vol, il):
                ax.plot(xs_, ts_, color="#ff2020", lw=2.2)
    ax.set_xlabel("Crossline")
    ax.set_ylabel("TWT (ms)")
    ax.set_title(title)
    if horizons:
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fault_cuts(fault, vol, il):
    """[(xl array, t array)] segments where a fault surface cuts an inline."""
    from seismic_tool.faults import section_segments_3d
    r = section_segments_3d(fault, vol, "Inline", il)
    if r is None:
        return []
    x0, y0, x1, y1 = (np.asarray(a, float) for a in r)
    return [(np.array([a, c]), np.array([b, d])) for a, b, c, d in zip(x0, y0, x1, y1)]


def vol_name_of(project, vol):
    for v in project.volumes:
        try:
            if project.volume(v["name"]) is vol:
                return v["name"]
        except Exception:
            continue
    return project.volumes[0]["name"]


# ============================================================================================ chapters
def ch_open(ctx):
    p, vol = ctx.project(), ctx.vol()
    ctx.n.update(vol=ctx.vol_name(), il0=int(vol.inlines[0]), il1=int(vol.inlines[-1]), xl0=int(vol.xlines[0]),
                 xl1=int(vol.xlines[-1]), ns=int(vol.ns), dt=f"{vol.dt * 1000:g}", t1=f"{float(vol.times[-1]):.0f}",
                 nh=len(p.horizons.items), hz=", ".join(h.name for h in p.horizons.items),
                 nw=len(p.wells()), wells=", ".join(w[0] for w in p.wells()),
                 faults=", ".join(f.name for f in p.faults.items) or "none", il=ctx.il_focus())
    xs, ys = vol.outline()
    ctx.n.update(x0=f"{min(xs):,.0f}", x1=f"{max(xs):,.0f}", y0=f"{min(ys):,.0f}", y1=f"{max(ys):,.0f}")
    from seismic_tool import maplayers as ML
    st = ML.LayerStack.load(p)

    def draw(path):
        ML.render_png(st, path, title=f"Basemap: {ctx.n['vol']} outline + {ctx.n['nw']} wells", size=(9, 7))
        return [(0.5, 0.5, "3D survey outline (rotated F3 bin grid)"), (0.2, 0.85, "Wells with names")]
    ctx.export("basemap", "Basemap of the project", draw)

    def sec(path):
        _section_png(path, vol, ctx.il_focus(), ctx.horizon_names(), p, f"Inline {ctx.il_focus()} with the "
                     f"interpreted horizons and faults")
        return [(0.5, 0.5, "Seismic, TWT down"), (0.85, 0.85, "Horizons (legend)")]
    ctx.export("inline", f"Inline {ctx.il_focus()}", sec)
    if ctx.mode == "real":
        mw = ctx.mw()
        mw.section.show("3D", ctx.vol_name(), "Inline", ctx.il_focus())
        ctx.window("main", "Seismic Interpretation main window", mw)


def ch_wellcorr(ctx):
    p = ctx.project()
    from well_correlation.project import Project
    from well_correlation.export import export_section
    wp = Project.open(p.well_project)
    names = list(wp.sections.get(wp.current_section) or sorted(wp.wells))
    tops = {w: (wp.tops or {}).get(w, {}) for w in names}
    fs6 = ctx.hz(["FS6"])
    d6 = [(w, tops[w].get(fs6)) for w in names if tops[w].get(fs6) is not None]
    ctx.n.update(wc_wells=", ".join(names), wc_section=wp.current_section or "-", n_fm=len(wp.formations),
                 n_tops=sum(len(t) for t in tops.values()),
                 fs6_md=", ".join(f"{w} {v:.0f} m" for w, v in d6) or "n/a")

    def draw(path):
        export_section(wp, names, path)
        return [(0.5, 0.04, "Well names + track headers"), (0.5, 0.5, "GR / DT / density tracks"),
                (0.45, 0.7, "F3 markers correlated between the wells")]
    ctx.export("wellsection", f"Well section '{ctx.n['wc_section']}'", draw)
    if ctx.mode == "real":
        from well_correlation.gui.main_window import MainWindow as WCMain
        ctx._wc = WCMain(p.well_project)
        ctx.window("wc_main", "Well Correlation main window", ctx._wc)


def ch_welltie(ctx):
    p = ctx.project()
    from seismic_tool.gui.welltie_dialog import WellTieDialog
    rows = []
    last = None
    for w, x, y in p.wells():
        dlg = WellTieDialog(p, ("3D", ctx.vol_name()), None, w)
        dlg.well_name = w
        dlg.freq = 30.0
        dlg.wavelet_kind = "Ricker"
        try:
            dlg.compute()
            res = dlg.auto()
            dlg.compute()
            c = dlg.tie_corr()
            rows.append((w, float(res.get("shift_ms", np.nan)), float(c if c is not None else res.get("corr", np.nan))))
            last = (dlg, None, w)
        except Exception as e:  # noqa: BLE001
            rows.append((w, np.nan, np.nan))
            ctx.log(f"    tie {w}: {e}")
    ctx.n["ties"] = "; ".join(f"{w}: shift {s:+.1f} ms, correlation {c:.2f}" for w, s, c in rows if np.isfinite(c)) or "n/a"
    ok = [c for _, _, c in rows if np.isfinite(c)]
    ctx.n["tie_best"] = f"{max(ok):.2f}" if ok else "n/a"

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, max(1, len(rows)), figsize=(12, 6.5), dpi=100, sharey=False)
        axs = np.atleast_1d(axs)
        for ax, (w, x, y) in zip(axs, p.wells()):
            dlg = WellTieDialog(p, ("3D", ctx.vol_name()), None, w)
            dlg.well_name, dlg.freq, dlg.wavelet_kind = w, 30.0, "Ricker"
            try:
                dlg.compute()
                dlg.auto()
                d = dlg.compute()
                syn = np.asarray(d["syn"], float)[3:-3]
                tt = np.asarray(dlg.tied_times(), float)[3:-3]  # synthetic samples at their tied seismic times
                ts, seis = np.asarray(d["ts"], float), np.asarray(d["seis"], float)
                m = (ts >= np.nanmin(tt)) & (ts <= np.nanmax(tt))
                ax.plot(syn / (np.nanmax(np.abs(syn)) or 1), tt, "b", lw=1, label="synthetic")
                ax.plot(seis[m] / (np.nanmax(np.abs(seis[m])) or 1) + 2.2, ts[m], "k", lw=1, label="seismic")
                ax.set_ylim(np.nanmax(tt), np.nanmin(tt))
                ax.set_title(f"{w}  r = {dlg.tie_corr():.2f}", fontsize=10)
            except Exception as e:  # noqa: BLE001
                ax.text(0.5, 0.5, f"{e}", transform=ax.transAxes, ha="center", fontsize=7, wrap=True)
                ax.set_title(w, fontsize=10)
            ax.set_xticks([])
        axs[0].set_ylabel("TWT (ms)")
        axs[0].legend(loc="lower left", fontsize=7)
        fig.suptitle("Well ties: synthetic (blue) vs seismic at the well (black) after Auto tie")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.12, 0.5, "Synthetic seismogram (blue)"), (0.2, 0.5, "Seismic trace at the well (black)")]
    ctx.export("welltie", "Well ties of the four wells", draw)
    if ctx.mode == "real" and last is not None:
        last[0].show()
        ctx.window("welltie_win", f"Well tie dialog ({last[2]})", last[0])


def ch_conditioning(ctx):
    mw = ctx.mw()
    vol = ctx.vol()
    il = ctx.il_focus()
    mw.section.show("3D", ctx.vol_name(), "Inline", il)
    dlg = mw.conditioning()
    try:
        dlg.load_preset("Interpretation")
    except Exception as e:  # noqa: BLE001
        ctx.log(f"    preset: {e}")
    t0, t1 = 1200.0, 1350.0
    if float(vol.times[-1]) < t1:
        t0, t1 = float(vol.times[len(vol.times) // 2]), float(vol.times[len(vol.times) // 2 + 30])
    dlg.add_zone(int(vol.inlines[0]), int(vol.inlines[-1]), int(vol.xlines[0]), int(vol.xlines[-1]), t0, t1, 0.0)
    q = dlg.preview("Inline", il)
    ctx.n.update(rms=f"{q['rms_removed_pct']:.0f}", leak=f"{q.get('leakage', float('nan')):.2f}",
                 chain=" > ".join(s["type"] for s in dlg.chain), pt0=f"{t0:.0f}", pt1=f"{t1:.0f}")
    last = dlg.last

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 3, figsize=(13, 5.6), dpi=100, sharey=True)
        t = np.asarray(vol.times, float)
        o = np.asarray(last["original"], float)
        c = np.nanpercentile(np.abs(o), 98) or 1.0
        for ax, k in zip(axs, ("original", "conditioned", "difference")):
            a = np.asarray(last.get(k, o - np.asarray(last["conditioned"], float)), float)
            ax.imshow(a.T, aspect="auto", cmap="seismic" if k != "difference" else "gray", vmin=-c, vmax=c,
                      extent=[vol.xlines[0], vol.xlines[-1], t[-1], t[0]])
            ax.axhline(t0, color="#ffcc00", lw=1, ls="--")
            ax.axhline(t1, color="#ffcc00", lw=1, ls="--")
            ax.set_title(k.capitalize())
            ax.set_xlabel("Crossline")
        axs[0].set_ylabel("TWT (ms)")
        fig.suptitle(f"Seismic > Conditioning, preset Interpretation, inline {il}: {ctx.n['rms']} % RMS removed; "
                     f"protect zone {t0:.0f}-{t1:.0f} ms (dashed)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.18, 0.5, "Original"), (0.5, 0.5, "Conditioned"), (0.82, 0.5, "Difference = what was removed"),
                (0.5, 0.42, "Protect zone (not filtered)")]
    ctx.export("conditioning", "Conditioning preview", draw)
    ctx.window("conditioning_win", "Seismic > Conditioning dialog", dlg)


def ch_picking(ctx):
    p = ctx.project()
    vol = ctx.vol()
    from seismic_tool.horizons import Picker, TrackParams
    src = ctx.hz(["FS6"])
    key = f"3D:{ctx.vol_name()}"
    ref = np.asarray(p.horizons.get(src).data[key], float)
    name = f"{src} (tutorial retrack)"
    old = p.horizons.get(name)
    if old is not None:
        p.horizons.items.remove(old)
    h = p.horizons.new(name)
    pk = Picker(p, p.horizons)
    il = ctx.il_focus()
    i = vol.il_index(il)
    # window around the focus inline (small, so the tutorial stays quick on the full F3 volume)
    span = max(4, min(40, vol.nil // 3))
    ii0, ii1 = max(0, i - span), min(vol.nil, i + span + 1)
    jj = np.arange(0, vol.nxl, max(1, vol.nxl // 12))
    seeds = [(i, int(j)) for j in jj if np.isfinite(ref[i, j])]
    pk.pick_points(h, key, [s[0] for s in seeds], [s[1] for s in seeds], [ref[s] for s in seeds], mode="Peak",
                   snap_ms=8.0)
    allowed = np.zeros(ref.shape, bool)
    allowed[ii0:ii1] = True
    t0 = time.time()
    pk.track(h, key, seeds, TrackParams(mode="Peak", window_ms=20, max_jump_ms=8, min_corr=0.6),
             faults=list(p.faults.items), allowed_mask=allowed)
    sec = time.time() - t0
    g = np.asarray(h.data[key], float)
    win = np.zeros(g.shape, bool)
    win[ii0:ii1] = True
    tracked = np.isfinite(g) & win
    diff = np.abs(g - ref)[tracked & np.isfinite(ref)]
    ctx.n.update(pick_hz=src, seeds=len(seeds), tracked=f"{tracked.sum() / max(1, win.sum()):.0%}",
                 pick_med=f"{np.median(diff):.1f}" if diff.size else "n/a", pick_s=f"{sec:.1f}",
                 il_a=int(vol.inlines[ii0]), il_b=int(vol.inlines[ii1 - 1]))

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=100)
        X = np.where(win, g, np.nan)[ii0:ii1]
        a = axs[0].imshow(X, aspect="auto", cmap="viridis_r", extent=[vol.xlines[0], vol.xlines[-1],
                                                                      vol.inlines[ii1 - 1], vol.inlines[ii0]])
        fig.colorbar(a, ax=axs[0], label="TWT (ms)")
        axs[0].set_title(f"Autotracked {src} from {len(seeds)} seeds (IL {ctx.n['il_a']}-{ctx.n['il_b']})")
        axs[0].set_xlabel("Crossline")
        axs[0].set_ylabel("Inline")
        D = np.where(win, g - ref, np.nan)[ii0:ii1]
        b = axs[1].imshow(D, aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8,
                          extent=[vol.xlines[0], vol.xlines[-1], vol.inlines[ii1 - 1], vol.inlines[ii0]])
        fig.colorbar(b, ax=axs[1], label="ms")
        axs[1].set_title(f"Retrack minus the interpreted {src} (median |diff| {ctx.n['pick_med']} ms)")
        axs[1].set_xlabel("Crossline")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.25, 0.5, "Tracked horizon (time)"), (0.75, 0.5, "Difference to the interpreted horizon")]
    ctx.export("tracking", "Autotracking from seeds", draw)
    p.horizons.items.remove(h)


def ch_faultmap(ctx):
    vol = ctx.vol()
    from seismic_tool import faultlik as FL
    il = ctx.il_focus()
    i = vol.il_index(il)
    half = min(6, max(2, vol.nil // 6))
    i0, i1 = max(0, i - half), min(vol.nil, i + half + 1)
    k0, k1 = 0, vol.ns
    if vol.ns > 300:
        k0, k1 = int(0.2 * vol.ns), int(0.9 * vol.ns)
    blk = np.asarray(vol.ilxl[i0:i1, :, k0:k1], np.float32) * vol.inv
    t0 = time.time()
    r = FL.fault_likelihood_block(blk, vol.dt * 1000.0)
    sec = time.time() - t0
    fl = r["fl"][i - i0]
    ctx.n.update(fl_box=f"IL {int(vol.inlines[i0])}-{int(vol.inlines[i1 - 1])}, {blk.shape[1]} crosslines, "
                        f"{float(vol.times[k0]):.0f}-{float(vol.times[k1 - 1]):.0f} ms",
                 fl_s=f"{sec:.1f}", fl_hi=f"{np.median(fl):.2f}")
    # where Fault A cuts the focus inline vs the likelihood maximum (median over the cut, metres)
    hit = "n/a"
    try:
        f = next((f for f in ctx.project().faults.items if "fault a" in f.name.lower()), None)
        cuts = fault_cuts(f, vol, il) if f is not None else []
        tw = np.asarray(vol.times, float)[k0:k1]
        bin_m = float(np.hypot(vol.ax[2], vol.ay[2]))
        dists, xf = [], []
        for xs_, ts_ in cuts:
            xm, tm = float(np.mean(xs_)), float(np.mean(ts_))
            if not (tw[0] <= tm <= tw[-1]):
                continue
            kk = int(np.argmin(np.abs(tw - tm)))
            jm = int(np.argmin(np.abs(vol.xlines - xm)))
            w = max(2, int(40 / max(1, vol.xl_step)))
            j0, j1 = max(0, jm - w), min(vol.nxl, jm + w + 1)
            jx = j0 + int(np.argmax(fl[j0:j1, kk]))
            dists.append(abs(float(vol.xlines[jx]) - xm) * bin_m)
            xf.append(xm)
        if dists:
            hit = f"{np.median(dists):.0f} m"
            ctx.n["fa_xl"] = f"{np.mean(xf):.0f}"
    except Exception as e:  # noqa: BLE001
        ctx.log(f"    fault A position: {e}")
    ctx.n["fl_hit"] = hit
    t = np.asarray(vol.times, float)[k0:k1]

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=100, sharey=True)
        s = blk[i - i0]
        c = np.nanpercentile(np.abs(s), 98) or 1.0
        axs[0].imshow(s.T, aspect="auto", cmap="gray", vmin=-c, vmax=c, extent=[vol.xlines[0], vol.xlines[-1], t[-1], t[0]])
        axs[0].set_title(f"Inline {il} (seismic)")
        axs[1].imshow(fl.T, aspect="auto", cmap="hot_r", vmin=0, vmax=1, extent=[vol.xlines[0], vol.xlines[-1], t[-1], t[0]])
        axs[1].set_title("Fault likelihood (0-1)")
        for ax in axs:
            ax.set_xlabel("Crossline")
        axs[0].set_ylabel("TWT (ms)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.25, 0.5, "Input inline"), (0.75, 0.5, "Fault likelihood: dark = likely fault")]
    ctx.export("faultlik", "Automated fault mapping - fault likelihood", draw)
    if ctx.mode == "real":
        ctx.window("faultmap_win", "Automated fault mapping dialog", ctx.mw().fault_mapping(ctx.vol_name()))


def _horizon_tau(ctx, il):
    """Horizon-guided relative geologic time on one inline: tau linear between the interpreted horizons (fast stand-in
    for Interpret > Compute RGT on a whole volume; the stack / Wheeler code is the same)."""
    p, vol = ctx.project(), ctx.vol()
    i = vol.il_index(il)
    key = f"3D:{ctx.vol_name()}"
    names = ctx.horizon_names()
    rows = []
    for n in names:
        rows.append(np.asarray(p.horizons.get(n).data[key], float)[i])
    H = np.array(rows)  # (K, nxl)
    order = np.argsort(np.nanmedian(H, axis=1))
    H, names = H[order], [names[k] for k in order]
    med = np.nanmedian(H, axis=1)
    t = np.asarray(vol.times, float)
    tau = np.empty((vol.nxl, vol.ns))
    for j in range(vol.nxl):
        ok = np.isfinite(H[:, j])
        xp = np.r_[t[0], H[ok, j], t[-1]]
        fp = np.r_[t[0], med[ok], t[-1]]
        o = np.argsort(xp)
        xp, fp = xp[o], np.maximum.accumulate(fp[o])
        tau[j] = np.interp(t, xp, fp)
    return tau, names, med


def ch_wheeler(ctx):
    vol = ctx.vol()
    il = ctx.il_focus()
    from seismic_tool.rgt.stack import wheeler
    tau, names, med = _horizon_tau(ctx, il)
    seis = np.asarray(vol.inline(il), float)
    t = np.asarray(vol.times, float)
    levels = np.linspace(med.min() - 100, min(med.max() + 100, t[-1]), 160)
    amp, hiatus = wheeler(seis, tau, levels, t)
    ctx.n.update(wh_levels=len(levels), wh_hiatus=f"{hiatus.mean():.1%}", wh_hz=", ".join(names))

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), dpi=100)
        c = np.nanpercentile(np.abs(seis), 98) or 1.0
        axs[0].imshow(seis.T, aspect="auto", cmap="seismic", vmin=-c, vmax=c, extent=[vol.xlines[0], vol.xlines[-1], t[-1], t[0]])
        cs = axs[0].contour(vol.xlines, t, tau.T, levels=levels[::8], colors="k", linewidths=0.4)
        axs[0].set_title(f"Inline {il} + RGT contours")
        axs[0].set_ylabel("TWT (ms)")
        A = np.where(hiatus, np.nan, amp)
        axs[1].imshow(A, aspect="auto", cmap="seismic", vmin=-c, vmax=c,
                      extent=[vol.xlines[0], vol.xlines[-1], levels[-1], levels[0]])
        axs[1].set_title("Wheeler (flattened on RGT); white = hiatus / condensed")
        axs[1].set_ylabel("Relative geologic time")
        for ax in axs:
            ax.set_xlabel("Crossline")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        del cs
        return [(0.25, 0.5, "RGT contours follow the reflectors"), (0.75, 0.5, "Wheeler: every level flat")]
    ctx.export("wheeler", "RGT contours and Wheeler diagram", draw)


def ch_velocity(ctx):
    p = ctx.project()
    from seismic_tool.velocity import VelocityModel
    from well_correlation.project import Project
    wp = Project.open(p.well_project)
    fms = {f["name"].lower(): f["name"] for f in wp.formations}
    pairs = [(n, fms[n.lower()]) for n in ctx.horizon_names() if n.lower() in fms]
    key = f"3D:{ctx.vol_name()}"
    meds = {n: float(np.nanmedian(p.horizons.get(n).data[key])) for n, _ in pairs}
    pairs = sorted(pairs, key=lambda q: meds[q[0]])
    pick = [q for q in pairs if q[0].upper() in ("TRUNCATION", "FS8", "FS6", "MFS4")] or pairs[:3]
    t0 = time.time()
    vm = VelocityModel(p, key, pick, residual_correction=True, cell=250.0)
    vm.build()
    sec = time.time() - t0
    mt = vm.misfit_table()
    res = np.array([r[5] for r in mt], float)
    k6 = next((k for k, (n, _) in enumerate(pick) if n.upper() == "FS6"), len(pick) - 1)
    g = vm.depth_grid(k6)
    ctx.n.update(vm_layers=", ".join(f"{n} = {f}" for n, f in pick), vm_s=f"{sec:.1f}",
                 vm_mis=f"{np.nanmax(np.abs(res)):.2f}" if res.size else "n/a", vm_hz=pick[k6][0],
                 vm_z0=f"{np.nanmin(g.z):.0f}", vm_z1=f"{np.nanmax(g.z):.0f}", vm_nw=len({r[0] for r in mt}))
    ctx.vm = vm
    ctx.vm_k = k6

    def draw(path):
        from seismic_tool import maps as M
        M.export_png(g, path, title=f"{pick[k6][0]} depth (m TVDSS) - layer-cake model, residual-corrected at the wells")
        return [(0.45, 0.5, "Depth map with contours"), (0.92, 0.5, "Colour bar (m)")]
    ctx.export("depthmap", f"{pick[k6][0]} depth map", draw)
    if ctx.mode == "real":
        ctx.window("velocity_win", "Velocity model + depth conversion dialog", ctx.mw().velocity_model())
        ctx.window("velvolume_win", "Velocity volume depth conversion (advanced) dialog", ctx.mw().velocity_volume())


def ch_livemap(ctx):
    mw = ctx.mw()
    lm = mw.live_map()
    fs6 = ctx.hz(["FS6"])
    t0 = time.time()
    lid = lm.add_live_grid(fs6, "horizon", horizon=fs6, key=f"3D:{ctx.vol_name()}")
    sec = time.time() - t0
    g = lm.canvas.stack.grid(lid)
    X, Y = g.node_xy()
    vol = ctx.vol()
    ox, oy = vol.outline()
    t = lm.bus.table
    xw = t.numeric("X")
    cut = float(np.round(np.nanmedian(xw), -3))
    lm.set_color("rules", rules=lm.parse_rules(f"X <= {cut:.0f} = lime; X > {cut:.0f} = steelblue"))
    lm.add_filter("X")
    lm.set_range_filter("X", None, float(np.nanmax(xw)) + 1)
    nvis = int(lm.styled["visible"].sum())
    ids = list(t.ids)
    lm.select_wells(ids[:2]) if hasattr(lm, "select_wells") else None
    opened = []
    real_open = lm.open_arbitrary
    lm.open_arbitrary = (lambda name, xs, ys: opened.append(name) or name) if ctx.mode != "real" else real_open
    lm.start_arbitrary()
    for k, (x, y) in enumerate([(float(np.min(ox)) + 800, float(np.mean(oy))), (float(np.mean(ox)), float(np.mean(oy)) + 1500),
                                (float(np.max(ox)) - 800, float(np.mean(oy)))]):
        lm.canvas.poly_click(x, y, "left", double=(k == 2))
    lm.open_arbitrary = real_open
    ctx.n.update(lg_hz=fs6, lg_s=f"{sec:.1f}", lg_x0=f"{float(X.min()):,.0f}", lg_x1=f"{float(X.max()):,.0f}",
                 out_x0=f"{float(min(ox)):,.0f}", out_x1=f"{float(max(ox)):,.0f}", lg_nodes=f"{g.z.shape[0]} x {g.z.shape[1]}",
                 rule=f"X <= {cut:.0f} = lime; X > {cut:.0f} = steelblue", nvis=nvis, nwells=len(ids))

    def draw(path):
        lm.export_map(path, title=f"Live Map - {fs6} Live grid", dpi=110)
        return [(0.36, 0.40, f"Live grid {fs6} (inside the survey outline)"), (0.71, 0.06, "Title block"),
                (0.71, 0.53, "Legend: one row per rule + other"), (0.71, 0.79, "Scale bar"), (0.8, 0.9, "North arrow")]
    ctx.export("livemap", "Live Map export", draw)
    lm.clear_filters()
    ctx.window("livemap_win", "Live Map window", lm)


def ch_glow(ctx):
    p, vol = ctx.project(), ctx.vol()
    from seismic_tool.glow import export_section_png, GlowSettings
    il = ctx.il_focus()
    i = vol.il_index(il)
    data = np.asarray(vol.inline(il), float)
    key = f"3D:{ctx.vol_name()}"
    curves = []
    cols = ["#ffd400", "#00e0ff", "#7CFC00", "#ff66cc", "#ffffff", "#ff9900", "#b0a0ff"]
    for k, n in enumerate(ctx.horizon_names()):
        z = np.asarray(p.horizons.get(n).data[key], float)[i]
        curves.append({"x": np.asarray(vol.xlines, float), "y": z, "colour": cols[k % len(cols)], "width": 2,
                       "kind": "horizon", "name": n})
    for f in p.faults.items:
        for xs_, ts_ in fault_cuts(f, vol, il):
            curves.append({"x": xs_, "y": ts_, "colour": "#ff2020", "width": 3, "kind": "fault", "name": f.name})
    n_glow = {}

    def draw_mode(mode):
        def draw(path):
            st = GlowSettings()
            try:
                st.settings["mode"] = mode
            except Exception:
                pass
            n_glow[mode] = export_section_png(path, data, np.asarray(vol.xlines, float), np.asarray(vol.times, float),
                                              curves, settings=st, with_glow=True, title=f"Inline {il} - {mode}",
                                              xlabel="Crossline")
            return [(0.5, 0.45, "Horizons and Fault A with a " + mode)]
        return draw
    ctx.export("glow", "Glow on horizons and faults", draw_mode("glow"))
    ctx.export("halo", "Halo on horizons and faults", draw_mode("halo"))
    ctx.n["glow_n"] = n_glow.get("glow", "n/a")
    from seismic_tool import aesthetics as AE
    fs6 = ctx.hz(["FS6"])
    z = np.asarray(p.horizons.get(fs6).data[key], float)

    def hill(path):
        plt = _plt()
        sh = AE.hillshade(np.nan_to_num(z, nan=np.nanmean(z)), 25.0 * vol.xl_step, 25.0 * vol.il_step, positive_down=True)
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=100)
        ax.imshow(z, cmap="viridis_r", origin="lower", aspect="auto")
        ax.imshow(sh, cmap="gray", origin="lower", aspect="auto", alpha=0.45)
        ax.set_title(f"{fs6} time map with hillshade (View > Sun / hillshade)")
        ax.set_xlabel("Crossline index")
        ax.set_ylabel("Inline index")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.5, 0.5, "Hillshade brings out the dome and Fault A")]
    ctx.export("hillshade", f"{fs6} hillshade", hill)


def ch_view3d(ctx):
    mw = ctx.mw()
    v = mw.viewer3d(ctx.vol_name(), force_software=True)
    vol = v.vol
    v.scene.fit()

    def draw(path):
        v.scene.render_png(path, size=(1200, 780), title="3D Viewer - F3: inline / crossline / time slice, horizons, "
                                                         "Fault A, wells")
        return [(0.5, 0.3, "Inline / crossline planes"), (0.4, 0.55, "Horizons"), (0.6, 0.6, "Fault A"),
                (0.55, 0.85, "Wells")]
    ctx.export("view3d", "3D Viewer (software renderer)", draw)
    box = ((max(0, vol.nil // 2 - 15), min(vol.nil, vol.nil // 2 + 15)), (0, vol.nxl),
           (int(0.3 * vol.ns), int(0.75 * vol.ns)))
    t0 = time.time()
    vr = v.volume_render(box=box, max_voxels=60000, n_slabs=12)
    # geobodies on the same box (the window's Geobodies button labels the whole volume - minutes on full F3)
    from seismic_tool import geobody as GB
    (a, b), (c, d), (e, f) = box
    sub = np.asarray(vol.ilxl[a:b], np.float32)[:, c:d, e:f] * vol.inv
    thr = float(np.percentile(sub, 99.0))
    lab, nb = GB.label_components(sub > thr)
    sizes = np.bincount(lab.ravel())[1:] if nb else np.zeros(0)
    ctx.n.update(vr_slabs=vr.get("slabs", 0), vr_step=vr.get("step", 1), n_bodies=int((sizes >= 20).sum()),
                 gb_thr=f"{thr:.2f}", v3_s=f"{time.time() - t0:.1f}",
                 gb_box=f"IL {int(vol.inlines[a])}-{int(vol.inlines[b - 1])}, {float(vol.times[e]):.0f}-"
                        f"{float(vol.times[f - 1]):.0f} ms")

    def draw2(path):
        v.scene.render_png(path, size=(1200, 780), title="3D interpretation: volume rendering of a sub-volume")
        return [(0.5, 0.5, "Volume-rendered sub-volume (opacity slabs)")]
    ctx.export("interp3d", "3D interpretation", draw2)
    try:
        v.bus.unsubscribe("view3d")
    except Exception:
        pass
    if ctx.mode == "real":
        ctx.window("view3d_win", "3D Viewer window", mw.viewer3d(ctx.vol_name()))


def ch_framework(ctx):
    mw = ctx.mw()
    dlg = mw.framework(ctx.vol_name())
    keep = [n for n in dlg.horizons if n.upper() in ("TRUNCATION", "FS8", "FS6", "MFS4")]
    if len(keep) >= 2:
        dlg.horizons = keep  # the dialog keeps them top to bottom (sorted by time, PLAN 43c)
    orig = dlg._points
    dlg._points = lambda name, max_points=700: orig(name, max_points)  # tutorial: fewer TPS points (speed)
    t0 = time.time()
    m = dlg.build(cell=500.0)
    vm = getattr(ctx, "vm", None)
    if vm is not None:
        dlg.depth_model_fn = lambda: vm
        try:
            dlg.depth()
        except Exception as e:  # noqa: BLE001
            ctx.log(f"    framework depth: {e}")
    q = dlg.make_grid([("proportional", 2)] * (len(dlg.model.grids) - 1))
    sec = time.time() - t0
    ctx.fw = dlg
    ctx.n.update(fw_hz=", ".join(dlg.horizons), fw_blocks=m.nblocks, fw_cells=q.get("cells", "n/a"),
                 fw_active=q.get("active", "n/a"), fw_domain=dlg.model.domain, fw_s=f"{sec:.1f}",
                 fw_neg=q.get("negative_thickness", "n/a"))
    blocks = m.blocks

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=100)
        X, Y = m.geom.nodes()
        a = axs[0].pcolormesh(X, Y, blocks, cmap="tab10", shading="auto")
        axs[0].set_title(f"Fault blocks ({m.nblocks}) - Fault A as a barrier")
        g0 = dlg.model.grids[0]
        b = axs[1].pcolormesh(X, Y, g0, cmap="viridis_r", shading="auto")
        fig.colorbar(b, ax=axs[1], label="m" if dlg.model.domain == "depth" else "ms")
        axs[1].set_title(f"{dlg.horizons[0]} framework surface ({dlg.model.domain})")
        for ax in axs:
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
        axs[0].set_ylabel("Y (m)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        del a
        return [(0.25, 0.5, "Fault blocks"), (0.75, 0.5, "Framework surface per block")]
    ctx.export("framework", "Structural framework", draw)
    ctx.window("framework_win", "Structural framework + 3D grid dialog", dlg)


def ch_properties(ctx):
    dlg = getattr(ctx, "fw", None)
    if dlg is None:
        raise RuntimeError("needs the framework chapter")
    from seismic_tool import propmodel as PM
    grid = dlg.grid
    Xc, Yc, Zc, top, bot = PM.cell_centres(grid)
    rng = np.random.default_rng(5)
    phi = np.clip(0.22 - 0.00002 * (Zc - np.nanmin(Zc)) + 0.02 * rng.standard_normal(Zc.shape), 0.02, 0.35)
    act = np.asarray(grid.actnum) > 0
    phi = np.where(act, phi, np.nan)
    names = list(dlg.horizons)
    kf = next((k for k, n in enumerate(names) if n.upper() == "FS6"), 0)
    zone = np.broadcast_to(np.asarray(grid.zone)[:, None, None], phi.shape)
    res = act & (zone == min(kf + 1, int(np.max(grid.zone))))  # reservoir = the zone below FS6 (illustration)
    owc = float(np.nanpercentile(Zc[res], 3)) if res.any() else 0.0  # illustration: 3 % of the zone above it
    rows = PM.volumetrics(grid, np.where(res, phi, 0.0), sw=np.full(phi.shape, 0.3), ntg=np.full(phi.shape, 0.7),
                          owc=owc)
    stoiip = PM.totals(rows, "stoiip_bbl")
    ctx.n["pm_zone"] = f"{names[kf]}-{names[kf + 1]}" if kf + 1 < len(names) else names[kf]
    ctx.n.update(pm_cells=int(act.sum()), pm_phi=f"{np.nanmean(phi):.3f}", pm_owc=f"{owc:.0f}",
                 pm_stoiip=f"{stoiip / 1e6:.1f}", pm_blocks=len({r["block"] for r in rows}))

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), dpi=100)
        k = phi.shape[0] // 2
        a = axs[0].imshow(phi[k], origin="lower", cmap="YlOrBr", aspect="auto")
        fig.colorbar(a, ax=axs[0], label="porosity")
        axs[0].set_title(f"Porosity, layer {k + 1} of {phi.shape[0]}")
        axs[1].hist(phi[np.isfinite(phi)], bins=30, color="#c2401f")
        axs[1].set_title("Porosity histogram (all active cells)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.25, 0.5, "Property in the 3D grid"), (0.75, 0.5, "Distribution")]
    ctx.export("properties", "Property model", draw)
    if ctx.mode == "real":
        ctx.window("propmodel_win", "Properties + volumetrics dialog", dlg.properties())


def ch_prospects(ctx):
    vm = getattr(ctx, "vm", None)
    if vm is None:
        raise RuntimeError("needs the velocity chapter (depth map)")
    from seismic_tool import migration as MG
    from seismic_tool import prospect as PR
    g = vm.depth_grid(ctx.vm_k)
    z = np.asarray(g.z, float)
    cell = abs(g.dx * g.dy) if g.kind == "xy" else 25.0 * 25.0
    sp = MG.spill_analysis(np.where(np.isfinite(z), z, np.nanmax(z)), cell)
    traps = [(c, d) for c, d in sp["traps"].items() if d.get("spill") is not None]
    traps.sort(key=lambda q: -float(q[1].get("capacity") or 0.0))
    if traps:
        c, d = traps[0]
        cap = float(d.get("capacity") or 0.0)
        mc = PR.volumetrics(n=5000, grv_m3=("lognormal", 0.7 * cap + 1.0, 1.3 * cap + 2.0), ntg=0.6, phi=0.25,
                            sw=0.35, fvf=1.15, rf=0.3, fluid="oil")
        p50 = float(mc["recoverable"]["p50"])
        ctx.n.update(n_traps=len(traps), trap_spill=f"{d['spill']:.0f}", trap_crest=f"{d['depth']:.0f}",
                     trap_col=f"{d['spill'] - d['depth']:.0f}", p50=f"{p50 / 1e6:.2f}",
                     p50_unit=str(mc.get("unit", "bbl")))
    else:
        ctx.n.update(n_traps=0, trap_spill="n/a", trap_crest="n/a", trap_col="n/a", p50="n/a")
    from seismic_tool import spacing as SPC
    wells = ctx.project().wells()

    def draw(path):
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=100)
        X, Y = g.node_xy()
        a = axs[0].pcolormesh(X, Y, z, cmap="viridis_r", shading="auto")
        drain = np.asarray(sp.get("drain"), float)
        if drain.shape == z.shape:
            for c_, d_ in traps[:5]:
                axs[0].contour(X, Y, (drain == c_).astype(float), levels=[0.5], colors="w", linewidths=0.8)
        for c_, d_ in traps[:5]:
            k = np.unravel_index(int(c_), z.shape)
            axs[0].plot(X[k], Y[k], "r^", ms=10)
        axs[0].set_title(f"Traps on the depth map ({len(traps)}), red = crests")
        fig.colorbar(a, ax=axs[0], label="m TVDSS")
        for w, x, y in wells:
            try:
                ex, ey = SPC.ellipse_polygon({"x": x, "y": y, "a": 600.0, "b": 300.0, "az": 45.0})
            except Exception:
                tt = np.linspace(0, 2 * np.pi, 60)
                ex, ey = x + 600 * np.cos(tt), y + 300 * np.sin(tt)
            axs[1].plot(ex, ey, "-", color="#4e79a7")
            axs[1].plot(x, y, "ko")
            axs[1].text(x, y, " " + w, fontsize=8)
        axs[1].set_aspect("equal")
        axs[1].set_title("Spacing: drainage ellipses around the wells")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [(0.25, 0.5, "Migration traps (fill-spill)"), (0.75, 0.5, "Drainage ellipses")]
    ctx.export("prospects", "Prospects, migration and spacing", draw)


CHAPTERS = [
    dict(id="f3_open", title="W1. Open F3 and look around", run=ch_open,
         purpose="Like opening a Kingdom project or a Petrel survey: the 3D volume, the interpreted horizons, Fault A "
                 "and the four wells, all in one place.",
         steps=["Double-click **GeoHarmonic - Seismic Interpretation** on the Desktop.",
                "**File > Open project...** (**Ctrl+O**) > `Documents\\F3 Netherlands\\F3 Netherlands.seisproj.json`.",
                "In the **Data** panel double-click **{vol}**; type **{il}** in the inline box and press Enter.",
                "Page through with **PgDown / PgUp**; switch **Inline / Crossline / Time slice** on the toolbar.",
                "The **Basemap** panel shows the survey outline and the wells; click inside the outline to jump to "
                "that inline."],
         results="Volume **{vol}**: inlines {il0}-{il1}, crosslines {xl0}-{xl1}, {ns} samples at {dt} ms (0-{t1} "
                 "ms). Outline X {x0}-{x1} m, Y {y0}-{y1} m (ED50 / UTM 31N). {nh} horizons ({hz}); {nw} wells "
                 "({wells}); faults: {faults}.",
         look=["Horizons sit on their reflectors along the whole inline.", "Wells plot inside the outline."]),
    dict(id="f3_wells", title="W2. Well correlation of the four wells", run=ch_wellcorr,
         purpose="The Well Correlation Tool is the Petrel well-section / Kingdom cross-section window: logs side by "
                 "side, hung on TVDSS or flattened on a top, tops correlated between wells.",
         steps=["Double-click **GeoHarmonic - Well Correlation**; **File > Open project...** > `F3 "
                "Netherlands.wcproj.json`.",
                "The section **{wc_section}** opens: {wc_wells}.",
                "Hang on **TVDSS**, then **View > Flatten on** FS6 to check the correlation.",
                "Drag a top to move it; **Ctrl+Z** undoes. The seismic tool and the Live Map follow (map bus)."],
         results="{n_fm} formations, {n_tops} tops in the section. FS6 MD: {fs6_md}.",
         look=["Each marker lines up with a log break (GR / density).", "Flattened on FS6 the shallower markers "
               "stay roughly parallel."]),
    dict(id="f3_ties", title="W3. Well ties", run=ch_welltie,
         purpose="Synthetic seismogram and bulk / stretch tie (Kingdom SynPAK, Petrel Well tie, HRS).",
         steps=["Seismic tool: **Interpret > Well tie (synthetic seismogram)...**.",
                "Pick the well, wavelet **Ricker 30 Hz**, press **Compute**, then **Auto tie**.",
                "Check the correlation, then **Save** - the tie is stored in the project and every window uses it."],
         results="{ties}. Best correlation {tie_best}.",
         look=["Synthetic and seismic peaks line up after Auto tie.", "Shift of a few ms; a large shift means a "
               "wrong checkshot or datum."]),
    dict(id="f3_cond", title="W4. Seismic conditioning (Interpretation preset)", run=ch_conditioning,
         purpose="Structure-oriented filtering and noise removal before picking - like Petrel's structural smoothing "
                 "or Kingdom's filters, with a protect zone that stays untouched.",
         steps=["Show inline **{il}**.", "**Seismic > Conditioning (spikes, bandpass, FX decon, SOF, footprint)...**.",
                "Preset **Interpretation** (chain: {chain}).",
                "**Add protect zone...**: whole survey, **{pt0}-{pt1} ms**.", "**Preview** > original | "
                "conditioned | difference; **Run on volume** writes a new volume."],
         results="{rms} % of the RMS removed on inline {il}; signal-leakage score {leak} (low = only noise removed).",
         look=["The difference panel shows noise, not reflectors.", "Inside the protect zone the difference is "
               "zero."]),
    dict(id="f3_pick", title="W5. Horizon / fault picking and autotracking", run=ch_picking,
         purpose="Seed picks and the wavefront autotracker (Kingdom's Hunt, Petrel's seeded 3D autotracking); Fault "
                 "A stops the tracker.",
         steps=["Horizons panel **New**, name it; press **P** (pick mode).",
                "Click the {pick_hz} reflector on inline {il} a few times, then **T** (**Interpret > Track horizon "
                "from seeds**).", "Faults: **N**, click down the fault, **Enter** ends the stick.",
                "**Ctrl+Z** undoes the last edit."],
         results="{seeds} seeds on inline {il}; tracked {tracked} of inlines {il_a}-{il_b} in {pick_s} s; median "
                 "difference to the interpreted {pick_hz}: {pick_med} ms.",
         look=["The tracked surface stops at Fault A instead of jumping across."]),
    dict(id="f3_faults", title="W6. Automated fault mapping", run=ch_faultmap,
         purpose="Fault likelihood + fault skins (PaleoScan AFE, Petrel Fault extraction).",
         steps=["**Interpret > Automated fault mapping (fault likelihood, auto faults, polygons)...**.",
                "Compute on a box first ({fl_box}), check thresholds, then the whole volume.",
                "**Extract** skins, filter by strike / dip, **Create faults** (sticks every 5 lines), **Polygons** "
                "for the map."],
         results="Box {fl_box}: {fl_s} s; median likelihood {fl_hi}. Along the interpreted Fault A (around "
                 "crossline {fa_xl}) the likelihood maximum is {fl_hit} from the fault on average.",
         look=["A sharp dark line along Fault A; noise stays light grey."]),
    dict(id="f3_rgt", title="W7. RGT and Wheeler diagram", run=ch_wheeler,
         purpose="Relative geologic time, horizon stack and Wheeler diagram (PaleoScan, Petrel Geobody/Stratal "
                 "slicing).",
         steps=["**Interpret > Compute RGT (current 3D volume)...** (on F3 about 10-20 min; the tutorial uses the "
                "interpreted horizons as RGT control on one inline to stay quick).",
                "**Interpret > Build horizon stack (3D)...**.",
                "Section toolbar domain **Wheeler**: every level is flat; **Stratal slice** mode for maps."],
         results="{wh_levels} levels between {wh_hz}; {wh_hiatus} of the Wheeler cells flagged as hiatus / "
                 "condensed.",
         look=["RGT contours follow the reflectors", "Gaps (white) where intervals thin out."]),
    dict(id="f3_depth", title="W8. Velocity model and depth conversion", run=ch_velocity,
         purpose="Layer-cake velocity model tied to the well tops (Kingdom Dynamic Depth Conversion, Petrel "
                 "velocity modelling); the velocity volume is the advanced mode.",
         steps=["**Interpret > Velocity model + depth conversion...**.",
                "Layers: {vm_layers}; function V0 per layer; tick **Residual correction**; **Build**.",
                "**Show depth map**; **Use** makes it the depth model of every window (section domain **Depth**).",
                "Advanced: **Interpret > Velocity volume depth conversion (advanced)...** > Build > Blind test."],
         results="Built in {vm_s} s on {vm_nw} wells; largest misfit at the wells after residual correction "
                 "{vm_mis} m. {vm_hz} depth {vm_z0}-{vm_z1} m TVDSS.",
         look=["Misfit at the wells ~0 m with residual correction on.", "No bull's-eyes around the wells."]),
    dict(id="f3_map", title="W9. Live Map: Live grid, styling, filters, selection, arbitrary line, export",
         run=ch_livemap,
         purpose="Kingdom-style live base map with Spotfire-style well styling, shared selection with every window.",
         steps=["**View > Live Map** (**Ctrl+Shift+M**).",
                "**Live grid...** > *Horizon: {lg_hz}* - gridded on the survey's own bins (rotated), coarse while "
                "picking, full on pause.",
                "Colour by **rules**: `{rule}` > **Apply**.", "Add a filter on X; drag the range.",
                "Lasso wells; the seismic basemap and the Well tool select them too.",
                "**Arbitrary line**: click points, **double-click** (or **Enter** / right-click) - the section "
                "opens; **Esc** cancels.", "**Export map (PNG / PDF)...**."],
         results="Live grid {lg_hz}: {lg_nodes} nodes in {lg_s} s, X {lg_x0}-{lg_x1} m = the outline (X "
                 "{out_x0}-{out_x1} m). {nvis} of {nwells} wells shown after the filter.",
         look=["The grid sits exactly inside the outline (fix 43d).", "Legend has one row per rule + other.",
               "Wells hidden by a filter have no labels."]),
    dict(id="f3_look", title="W10. Glow, halo and aesthetics", run=ch_glow,
         purpose="Display polish: glowing interpretation, hillshade, themes and figure export.",
         steps=["**Display > Glow mode: Glow** (or **G**), **Display > Glow mode: Halo**.",
                "**View > Theme > Dark / Light**.", "**File > Export section figure (title block, scale bar, "
                "legend)...**."],
         results="{glow_n} glowing curves drawn on inline {il}.",
         look=["Thin horizons stay readable on busy seismic."]),
    dict(id="f3_3d", title="W11. 3D Viewer and 3D interpretation", run=ch_view3d,
         purpose="Petrel-style 3D window: planes, horizons, faults, wells; volume rendering and geobodies.",
         steps=["**View > 3D Viewer (slices, horizons, faults, wells)...** (**Ctrl+Shift+3**).",
                "Drag the plane sliders; left-drag orbits, wheel zooms.",
                "Toolbar **Volume render** on a box; **Geobodies** with a threshold."],
         results="Volume rendering of {gb_box}: {vr_slabs} slabs (read stride {vr_step}); {n_bodies} bodies of 20+ "
                 "cells above the 99th-percentile amplitude ({gb_thr}) in that box; {v3_s} s.",
         look=["Planes, horizons and Fault A line up in 3D."]),
    dict(id="f3_frame", title="W12. Structural framework and 3D grid", run=ch_framework,
         purpose="Fault-block framework and a stair-stepped corner-point grid (Petrel Structural framework / "
                 "Pillar gridding) exported as GRDECL.",
         steps=["**Interpret > Structural framework + 3D grid (GRDECL)...**.",
                "**1. Framework (time)**, **2. Depth (current depth model)**, **3. 3D grid**, **Export GRDECL...**."],
         results="Horizons {fw_hz}: {fw_blocks} fault blocks; grid {fw_cells} cells ({fw_active} active, negative "
                 "thickness {fw_neg}) in {fw_domain}; {fw_s} s.",
         look=["Two blocks either side of Fault A.", "No negative-thickness cells."]),
    dict(id="f3_props", title="W13. Property modelling and volumetrics", run=ch_properties,
         purpose="Properties in the 3D grid and volumetrics per zone and block (Petrel Property modelling).",
         steps=["In the framework dialog **4. Properties + volumetrics...**.",
                "**Upscale logs**, **Variogram**, **Simulate** (SGS), **Volumetrics** with the contacts, "
                "**Export**."],
         results="{pm_cells} active cells; mean porosity {pm_phi}. Zone {pm_zone} with an OWC at {pm_owc} m "
                 "(placed so 3 % of the zone's cells are above it - an illustration, not an F3 contact): STOIIP {pm_stoiip} MMbbl.",
         look=["Porosity follows the layering; no values in inactive cells."]),
    dict(id="f3_prospects", title="W14. Prospects, migration and spacing", run=ch_prospects,
         purpose="Fill-spill migration on the depth map, Rose-style volumetrics, drainage ellipses.",
         steps=["**Map layers > Add... > Migration: flow paths, traps, fill-spill on the selected depth grid**.",
                "**Map layers > Add... > DHI checklist...** stores a prospect (risk + volumetrics).",
                "Dashboard: **Spacing + ellipses...**."],
         results="{n_traps} traps; largest: crest {trap_crest} m, spill {trap_spill} m (column {trap_col} m); P50 "
                 "recoverable {p50} million {p50_unit} (illustrative parameters).",
         look=["Traps sit on the closures of the depth map."]),
]


def run_all(ctx, chapters=None, progress=None):
    """Run every chapter; returns [dict(id, title, purpose, steps(text filled), results, look, shots, error,
    seconds)]."""
    out = []
    chapters = chapters or CHAPTERS
    try:
        ctx.n.setdefault("il", ctx.il_focus())
        ctx.n.setdefault("vol", ctx.vol_name())
    except Exception as e:  # noqa: BLE001
        ctx.log(f"F3 project could not be opened: {e}")
    for k, c in enumerate(chapters, 1):
        ctx.shots = []
        t0 = time.time()
        err = None
        msg = f"[F3 {k}/{len(chapters)}] {c['title']} ..."
        (progress or ctx.log)(msg)
        try:
            c["run"](ctx)
        except Exception as e:  # noqa: BLE001 - captioned, the rest continues
            err = f"{type(e).__name__}: {e}"
            ctx.log(traceback.format_exc())
        sec = time.time() - t0
        (progress or ctx.log)(f"    {'ok' if err is None else 'FAILED: ' + err} ({sec:.1f} s, {len(ctx.shots)} pictures)")
        out.append(dict(id=c["id"], title=c["title"], purpose=c["purpose"], steps=[_fill(s, ctx.n) for s in c["steps"]],
                        results=_fill(c["results"], ctx.n), look=list(c.get("look", [])), shots=list(ctx.shots),
                        error=err, seconds=sec))
    return out


class _Missing(dict):
    def __missing__(self, k):
        return "n/a"


def _fill(text, n):
    try:
        return text.format_map(_Missing({k: v for k, v in n.items()}))
    except Exception:
        return text
