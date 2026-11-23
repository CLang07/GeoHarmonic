"""Exports: cross section to PNG/PDF (matplotlib), tops to CSV / OpendTect / Kingdom ASCII."""
import csv
import numpy as np

from .section import SectionLayout


def section_figure(project, well_names, display=None, ylim=None, height_in=11.0, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches  # noqa: F401
    import matplotlib.pyplot as plt

    lay = SectionLayout(project, well_names, display)
    if not lay.panels:
        raise ValueError("No wells in the section")
    width_in = max(6.0, 0.9 * lay.total_width + 1.5)
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes([0.07, 0.04, 0.9, 0.82])
    hx = fig.add_axes([0.07, 0.87, 0.9, 0.11], sharex=ax)
    hx.axis("off")
    lo, hi = ylim or lay.depth_range()
    for img, x0, x1, y0, y1, kind in lay.lith_images((lo, hi)):
        ax.imshow(img, extent=(x0, x1, y1, y0), aspect="auto", interpolation="nearest", zorder=0)
    for p in lay.panels:
        for xa, xb, y, col in lay.fills(p):
            ax.fill_betweenx(y, xa, xb, color=col, linewidth=0, alpha=0.85)
        for c in lay.curves(p):
            ax.plot(c["x"], c["y"], color=c["color"], linewidth=c["width"] * 0.5)
        for tid, tx0, tx1 in p.tracks:
            ax.axvline(tx0, color="#999999", lw=0.4)
            ax.axvline(tx1, color="#999999", lw=0.4)
        w = project.wells[p.name]
        hx.text((p.x0 + p.x1) / 2, 0.95, p.name, ha="center", va="top", fontsize=10, weight="bold")
        hx.text((p.x0 + p.x1) / 2, 0.72, f"{w.elev_ref} {w.elev:.1f} m", ha="center", va="top", fontsize=7)
        for tx0, tx1, title, curves in lay.headers(p):
            for k, (lab, l, r, col) in enumerate(curves):
                yy = 0.45 - k * 0.2
                if l != "":
                    hx.text(tx0 + 0.02, yy, f"{l:g}", fontsize=5, color=col, va="center")
                    hx.text(tx1 - 0.02, yy, f"{r:g}", fontsize=5, color=col, va="center", ha="right")
                hx.text((tx0 + tx1) / 2, yy, lab.split(" ")[0], fontsize=5.5, color=col, va="center", ha="center")
    from .lithology import CODES
    if "LITH" in lay.track_ids or lay.d.get("between") == "lithology":
        for k, (name, col) in CODES.items():
            hx.add_patch(matplotlib.patches.Rectangle((0.6 * k * lay.tw, -0.25), 0.15 * lay.tw, 0.12, color=col,
                                                      transform=hx.transData, clip_on=False))
            hx.text(0.6 * k * lay.tw + 0.18 * lay.tw, -0.19, name, fontsize=6, va="center")
    for i, dkm in enumerate(lay.distances):
        if dkm is not None:
            a, b = lay.panels[i], lay.panels[i + 1]
            hx.text((a.x1 + b.x0) / 2, 0.2, f"{dkm:.1f} km", ha="center", fontsize=6, color="#555555")
    for t in lay.top_marks():
        for p, y, md in t["marks"]:
            ax.plot([p.x0, p.x1], [y, y], color=t["color"], lw=1.2)
        for (x0, y0), (x1, y1) in t["segments"]:
            ax.plot([x0, x1], [y0, y1], color=t["color"], lw=0.8, ls="--")
        if t["marks"]:
            p, y, md = t["marks"][-1]
            ax.text(p.x1 + 0.05, y, t["name"], color=t["color"], fontsize=6, va="center")
    ax.set_ylim(hi, lo)
    ax.set_xlim(-0.2, lay.total_width + 0.2)
    ax.set_xticks([])
    ax.set_ylabel(lay.y_label())
    ax.grid(axis="y", color="#dddddd", lw=0.4)
    return fig


def export_section(project, well_names, path, display=None, ylim=None):
    fig = section_figure(project, well_names, display, ylim)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


def export_tops_csv(project, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Well", "UWI", "Formation", "MD_m", "TVDSS_m", "X_NZTM", "Y_NZTM"])
        for wn, tops in project.tops.items():
            well = project.wells.get(wn)
            if not well:
                continue
            for fm, md in sorted(tops.items(), key=lambda kv: kv[1]):
                z = float(well.md_to_tvdss([md])[0]) if well.loaded else md - (well.elev or 0)
                w.writerow([wn, well.uwi, fm, f"{md:.2f}", f"{z:.2f}", well.x and f"{well.x:.1f}", well.y and f"{well.y:.1f}"])
