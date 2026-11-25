"""3D interpretation tests (PLAN 34): geobodies (labelling, out-of-core, surfaces, footprints), volume rendering,
3D picking + tracking from 3D seeds, attribute volumes on the planes."""
import os

import numpy as np

from . import geobody as GB


def run(check, tmp):
    n = 40
    I, J, K = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    R = 12.0
    sph = (I - 20) ** 2 + (J - 20) ** 2 + (K - 20) ** 2 <= R * R
    V, T = GB.surface_nets(sph)
    cnt = GB.edge_use_counts(T)
    E = len(cnt)
    r = np.linalg.norm(V - 20.0, axis=1)
    check(np.all(cnt == 2) and len(V) - E + len(T) == 2 and abs(r.mean() - R) < 0.8 and r.max() - r.min() < 2.0,
          f"34d: closed surface of a sphere (every edge in 2 triangles, Euler 2, radius {r.mean():.1f} vs {R})")
    m = np.zeros((30, 30, 30), bool)
    m[2:8, 2:8, 2:8] = True
    m[15:25, 15:20, 10:28] = True
    m[9, 9, 9] = True  # diagonal neighbour of the first box: not 6-connected
    lab, nb = GB.label_components(m)
    sizes = np.bincount(lab.ravel())[1:]
    check(nb == 3 and sizes.tolist() == [900, 216, 1], "34d: two known bodies labelled with exact cell counts")
    V2, T2 = GB.surface_nets(np.ones((4, 4, 4), bool))
    check(np.all(GB.edge_use_counts(T2) == 2), "34d: a body touching the volume edge still gets a closed surface")
    hx, hy = GB.convex_hull(np.array([0, 2, 2, 0, 1]), np.array([0, 0, 2, 2, 1]))
    check(len(hx) == 4, "34d: convex hull outline")
    curve = [(-1.0, 0.8), (-0.2, 0.0), (0.2, 0.0), (1.0, 0.8)]
    check(np.allclose(GB.opacity_curve(np.array([-1.0, 0.0, 0.6]), curve), [0.8, 0.0, 0.4]), "34c: opacity curve")
    check(GB.integrated_graphics("Intel(R) UHD Graphics 620") and not GB.integrated_graphics("NVIDIA RTX A2000"),
          "34c: integrated-graphics warning")


def run_volume(check, tmp):
    """Out-of-core labelling on a stored volume (bodies crossing the inline blocks) + footprints + slabs."""
    from .project import SeisProject
    from .synth import write_3d
    from .volume import import_3d
    from .gui.colormaps import lut
    folder = os.path.join(tmp, "geob")
    os.makedirs(folder, exist_ok=True)
    nil, nxl, ns = 36, 30, 60
    cube = np.zeros((nil, nxl, ns), np.float32) + 0.01
    I, J, K = np.meshgrid(np.arange(nil), np.arange(nxl), np.arange(ns), indexing="ij")
    body1 = ((I - 17) / 12.0) ** 2 + ((J - 10) / 6.0) ** 2 + ((K - 20) / 5.0) ** 2 <= 1.0  # crosses block borders
    body2 = (I >= 4) & (I < 8) & (J >= 22) & (J < 27) & (K >= 40) & (K < 50)
    cube[body1] = 1.0
    cube[body2] = 0.8
    sgy = os.path.join(folder, "g.sgy")
    write_3d(sgy, cube, dt_us=4000)
    store = os.path.join(folder, "g_seismic", "G")
    import_3d(sgy, store)
    sp = SeisProject()
    sp.path = os.path.join(folder, "g.seisproj.json")
    sp.volumes.append({"name": "G", "store": store, "segy": sgy})
    vol = sp.volume("G")
    lab, bodies = GB.label_volume(vol, lo=0.5 * float(np.max(np.asarray(vol.ilxl[:]) * vol.inv)), block=7,
                                  out_path=os.path.join(folder, "labels.npy"))
    check(len(bodies) == 2 and bodies[0]["cells"] == int(body1.sum()) and bodies[1]["cells"] == int(body2.sum()) and
          abs(bodies[1]["volume_m3"] - body2.sum() * 25 * 25 * 5.0) < 1e-6,
          f"34d: out-of-core labelling across inline blocks ({[b['cells'] for b in bodies]} cells, volumes in m3)")
    fp = GB.footprint(lab, 1, vol, bodies[0]["bbox"])
    from .maphub import point_in_polygon
    cx, cy = vol.xy(vol.inlines[17], vol.xlines[10])
    check(len(fp["x"]) >= 4 and point_in_polygon(np.array([float(cx)]), np.array([float(cy)]), fp["x"], fp["y"])[0],
          "34d: body footprint polygon for the Live Map")
    mesh = GB.body_mesh_world(lab, 2, vol, bbox=bodies[1]["bbox"])
    check(len(mesh) > 0 and mesh.kind == "triangles", "34d: body surface in world coordinates")
    sub = np.asarray(vol.ilxl[:], np.float32) * vol.inv
    sl = GB.render_slabs(sub, (0, 0, 0), vol, lut("Grey"), (0, 1), [(0.0, 0.0), (0.3, 0.0), (1.0, 0.9)], n_slabs=12)
    alphas = np.concatenate([m.rgba[:, 3] for m in sl]) if sl else np.zeros(0)
    check(0 < len(sl) <= 12 and alphas.max() > 0.5 and alphas.min() >= 0, "34c: volume rendering as opacity slabs "
          "(transparent where the curve is 0)")
    return sp


def run_gui(check, tmp):
    """34 GUI: pick on a 3D plane into the horizon / fault objects, track from 3D seeds, volume rendering, geobodies,
    another volume on the planes, horizon coloured by an attribute."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    from . import scene3d as S3
    proj, truth, d = _setup(tmp, "interp3d")
    mw = MainWindow(proj.path)
    v = mw.viewer3d("Synth", force_software=True)
    vol = v.vol
    for mode in ("Crossline", "Time slice"):  # keep only the inline plane for a clean pick
        v.scene.set_visible(v.scene.find("slice", mode).id, False)
    il = int(vol.inlines[10])
    v.set_plane("Inline", il, publish=False)
    cam = S3.Camera(distance=2500.0, azimuth=180.0, elevation=5.0)  # inlines run E-W here: look north at it
    cam.target = S3.plane_corners(vol, "Inline", il).mean(axis=0)
    v.scene.camera = cam
    j = 12
    t_true = float(truth["3D:Synth"][10, j])
    x, y = vol.xy(float(il), float(vol.xlines[j]))
    px = cam.project([[float(x), float(y), S3.world_z(t_true + 3.0)]], 800, 600)[0]
    hit = v.pick(px[0], px[1], 800, 600)
    check(hit is not None and hit[0] == "Inline" and abs(hit[2] - vol.xlines[j]) < 0.6 and abs(hit[3] - t_true - 3) < 1.5,
          f"34a GUI: ray-plane pick -> {hit}")
    hz = mw.project.horizons.new("3D pick")
    res = v.pick_horizon("3D pick", px[0], px[1], 800, 600)
    check(res is not None and abs(res[2] - t_true) < 3.0 and np.isfinite(hz.data["3D:Synth"][10, j]),
          f"34a GUI: pick snapped to the reflector ({res}) in the 2D viewer's horizon object")
    n_before = len(mw.project.horizons.undo_stack)
    n = v.track("3D pick")
    g = hz.data["3D:Synth"]
    ok = np.isfinite(g) & np.isfinite(truth["3D:Synth"])
    check(n and ok.mean() > 0.8 and np.median(np.abs(g[ok] - truth["3D:Synth"][ok])) < 2.0 and
          len(mw.project.horizons.undo_stack) > n_before, f"34b GUI: tracked from the 3D seed over {ok.mean():.0%} of "
          "the survey (= item-1 tracker), undo available")
    mw.project.horizons.undo()
    check(np.isfinite(hz.data["3D:Synth"]).sum() <= 2, "34b GUI: one undo removes the tracking run")
    # fault stick picked in 3D
    for tt in (300.0, 500.0):
        p2 = cam.project([[float(x), float(y), S3.world_z(tt)]], 800, 600)[0]
        v.pick_fault_point(p2[0], p2[1], 800, 600)
    f = v.end_stick("3D fault")
    check(f is not None and len(f.sticks) == 1 and len(f.sticks[0]) == 2 and "3D fault" in mw.project.faults.names(),
          "34a GUI: fault stick picked on a 3D plane into the fault list")
    vr = v.volume_render(box=((0, vol.nil), (0, vol.nxl), (40, 140)), max_voxels=20000, n_slabs=10)
    check(vr["slabs"] > 0 and vr["step"] > 1 and v.scene.find("volume") is not None,
          f"34c GUI: volume rendering of a boxed sub-volume within the voxel budget (stride {vr['step']})")
    bodies = v.geobodies(lo=0.5, min_cells=20)
    lay = next((l for l in mw.basemap.stack.layers if l["name"].startswith("Geobodies")), None)
    check(bodies and lay is not None and v.scene.find("geobody") is not None,
          f"34d GUI: {len(bodies)} geobodies as 3D surfaces + outlines on the Live Map")
    from .attributes import compute_volume
    an = compute_volume(mw.project, "Synth", "Envelope")
    v.set_volume(an)
    check(v.vol_name == an and v.lut_name.startswith("Rainbow") and
          v.scene.find("slice", "Inline").meshes[0] is not None, "34e GUI: attribute volume shown on the 3D planes")
    v.set_volume("Synth")
    amp = np.abs(truth["3D:Synth"] - truth["3D:Synth"].mean())
    m = v.colour_horizon_by("Top", amp, "amplitude")
    check(len(m) > 0 and v.scene.find("horizon", "Top").colorbar[3] == "amplitude",
          "34e GUI: horizon attribute extraction shown on the 3D surface")
    mw.maplink.close()
