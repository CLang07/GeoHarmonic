"""Tests for the petrophysics package (PLAN item 20).  Called from selftest.run (check, tmp)."""
import json
import os

import numpy as np


def write_las(path, well, depth, curves, units=None, srvc="", extra_well=None, descs=None, null=-999.25):
    units = units or {}
    descs = descs or {}
    with open(path, "w") as f:
        f.write("~Version\n VERS. 2.0 : test\n WRAP. NO :\n~Well\n")
        f.write(f" STRT.M {depth[0]} :\n STOP.M {depth[-1]} :\n STEP.M {depth[1] - depth[0]:.4f} :\n NULL. {null} :\n")
        f.write(f" WELL. {well} : WELL\n SRVC. {srvc} : SERVICE COMPANY\n")
        for k, v in (extra_well or {}).items():
            f.write(f" {k}. {v} : {k}\n")
        f.write("~Curve\n DEPT.M : depth\n")
        for n in curves:
            f.write(f" {n}.{units.get(n, '')} : {descs.get(n, n)}\n")
        f.write("~A\n")
        arr = np.column_stack([depth] + [np.where(np.isfinite(v), v, null) for v in curves.values()])
        np.savetxt(f, arr, fmt="%.4f")


def layered_earth(top, base, rng, mean_thick=2.0, min_thick=0.25, thin_at=None):
    """Bed boundaries + a sand flag per bed.  thin_at: depth where a 0.3 m sand is inserted."""
    b = [top]
    while b[-1] < base:
        b.append(b[-1] + max(min_thick, rng.exponential(mean_thick)))
    b = np.array(b)
    sand = (np.arange(len(b) - 1) % 2 == 0)
    if thin_at is not None:
        j = np.searchsorted(b, thin_at)
        b = np.concatenate([b[:j], [thin_at, thin_at + 0.3], b[j:]])
        b = np.unique(b)
        sand = np.array([(k % 2 == 0) for k in range(len(b) - 1)])
        k = np.searchsorted(b, thin_at)
        sand[k] = True
        if k > 0:
            sand[k - 1] = False
        if k + 1 < len(sand):
            sand[k + 1] = False
    return b, sand


def sample_earth(b, sand, z, sand_val, shale_val, res, fine=0.01):
    """Log = bed property averaged over a box of the tool resolution, sampled at z."""
    zf = np.arange(z[0] - res, z[-1] + res, fine)
    j = np.clip(np.searchsorted(b, zf, side="right") - 1, 0, len(sand) - 1)
    prop = np.where(sand[j], sand_val, shale_val)
    n = max(1, int(round(res / fine)))
    k = np.ones(n) / n
    sm = np.convolve(prop, k, "same")
    return np.interp(z, zf, sm)


def run_qc(check, tmp):
    from ..project import Project
    from . import tools as T
    from . import qc as Q
    rng = np.random.default_rng(20)

    # ---------------- primitives
    z = np.arange(1000.0, 1600.0, 0.1524)
    v = 50 + 5 * rng.standard_normal(z.size)
    spikes_at = rng.choice(np.arange(50, z.size - 50), 6, replace=False)
    v[spikes_at] += 80
    sm = Q.spike_mask(v)
    check(sm[spikes_at].all() and sm.sum() <= len(spikes_at) + 2, f"spikes: {sm[spikes_at].sum()}/6 found, {sm.sum()} flagged")
    g = v.copy()
    g[(z >= 1300) & (z < 1305)] = np.nan
    gp = Q.gaps(z, g, 0.5)
    check(len(gp) == 1 and abs(gp[0][0] - 1300) < 0.2 and abs(gp[0][1] - 1305) < 0.3, f"gap found {gp}")
    f = v.copy()
    f[(z >= 1200) & (z < 1210)] = 2.45
    fm = Q.flatline_mask(z, f)
    iv = Q.intervals(fm, z)
    check(len(iv) == 1 and abs(iv[0][0] - 1200) < 0.2 and abs(iv[0][1] - 1210) < 0.3, f"flat line found {iv}")
    cali = 8.5 + 0.1 * rng.standard_normal(z.size)
    wash = (z >= 1100) & (z < 1120)
    cali[wash] += 3.0
    bh, bs, src = Q.bad_hole(z, cali)
    iv = Q.intervals(bh, z)
    check(np.nanmedian(bs) == 8.5 and len(iv) == 1 and abs(iv[0][0] - 1100) < 0.2 and abs(iv[0][1] - 1120) < 0.3,
          f"bad hole from caliper (bit size {np.nanmedian(bs)} estimated) {iv}")
    drho = np.zeros(z.size)
    drho[(z >= 1400) & (z < 1402)] = 0.3
    bh2, _, _ = Q.bad_hole(z, None, drho=drho)
    check(abs(bh2.sum() * 0.1524 - 2.0) < 0.31, "bad hole from |DRHO|")

    # depth shift: run 2 logged 1.37 m deep
    b, sand = layered_earth(990, 1610, rng)
    gr_true = lambda zz: sample_earth(b, sand, zz, 30.0, 110.0, 0.45)  # noqa: E731
    z2 = np.arange(1200.0, 1500.0, 0.1524)
    ref = gr_true(z) + 2 * rng.standard_normal(z.size)
    run2 = gr_true(z2 - 1.37) * 1.1 + 5 + 2 * rng.standard_normal(z2.size)  # different calibration too
    sh, cc, ov = Q.depth_shift(z, ref, z2, run2, 5.0)
    check(abs(sh + 1.37) < 0.08 and cc > 0.8, f"depth shift recovered {sh:+.2f} m (true -1.37), corr {cc:.2f}")

    # bed boundaries on a well-sampled high-resolution curve
    zh = np.arange(1000.0, 1600.0, 0.0508)
    pe = sample_earth(b, sand, zh, 2.0, 3.5, 0.1) + 0.03 * rng.standard_normal(zh.size)
    bnd = Q.bed_boundaries(zh, pe, smooth_m=0.05, min_sep_m=0.1)
    tb = b[(b > 1001) & (b < 1599)]
    d_true = np.min(np.abs(tb[:, None] - bnd[None, :]), axis=1)
    d_found = np.min(np.abs(bnd[:, None] - tb[None, :]), axis=1)
    check((d_true < 0.15).mean() >= 0.95 and (d_found < 0.15).mean() >= 0.9,
          f"bed boundaries: {100 * (d_true < 0.15).mean():.0f}% of {len(tb)} found, {100 * (d_found < 0.15).mean():.0f}% "
          f"of {len(bnd)} picks true")
    th = Q.bed_thickness_curve(zh, bnd)
    k = np.searchsorted(zh, 0.5 * (tb[3] + tb[4]))
    check(abs(th[k] - (tb[4] - tb[3])) < 0.2, "bed thickness curve")

    # ---------------- synthetic project: SLB wireline (2 runs, shifted), SLB LWD, Baker wireline
    root = os.path.join(tmp, "petro_proj")
    rng = np.random.default_rng(7)
    b, sand = layered_earth(990, 1710, rng, thin_at=1250.0)

    def logs(zz, shift=0.0):
        q = zz - shift
        out = {"GR": sample_earth(b, sand, q, 30.0, 110.0, 0.45) + 2 * rng.standard_normal(zz.size),
               "RHO": sample_earth(b, sand, q, 2.25, 2.55, 0.45) + 0.01 * rng.standard_normal(zz.size),
               "PE": sample_earth(b, sand, q, 1.9, 3.3, 0.1) + 0.03 * rng.standard_normal(zz.size),
               "NPH": sample_earth(b, sand, q, 0.22, 0.35, 0.6) + 0.005 * rng.standard_normal(zz.size),
               "DT": sample_earth(b, sand, q, 95.0, 120.0, 1.07) + 0.5 * rng.standard_normal(zz.size),
               "RD": sample_earth(b, sand, q, 20.0, 2.0, 0.45) * np.exp(0.02 * rng.standard_normal(zz.size)),
               "RS": sample_earth(b, sand, q, 8.0, 1.8, 0.45) * np.exp(0.02 * rng.standard_normal(zz.size))}
        return out

    z1 = np.arange(1000.0, 1500.0, 0.1524)
    a = logs(z1)
    cal = 8.5 + 0.05 * rng.standard_normal(z1.size)
    cal[(z1 >= 1100) & (z1 < 1110)] += 3.5  # washout (noisy: constant values would be blanked as a flat line)
    dtco = a["DT"].copy()
    dtco[[500, 1200, 2000]] += 60
    d = os.path.join(root, "wells", "Synth-1", "las")
    os.makedirs(d)
    write_las(os.path.join(d, "PEX-HALS-DSI_R1.las"), "Synth-1", z1,
              {"GR": a["GR"], "HLLD": a["RD"], "HLLS": a["RS"], "RHOZ": a["RHO"], "PEFZ": a["PE"], "TNPH": a["NPH"],
               "DTCO": dtco, "HCAL": cal, "BS": np.full(z1.size, 8.5), "TENS": np.full(z1.size, 3000.0)},
              {"GR": "GAPI", "HLLD": "OHMM", "TNPH": "V/V", "DTCO": "US/F", "HCAL": "IN"}, srvc="Schlumberger",
              extra_well={"RUN": "1"})
    z2 = np.arange(1380.0, 1700.0, 0.1524)
    a2 = logs(z2, shift=1.37)  # second run logged 1.37 m deep
    write_las(os.path.join(d, "PEX-HALS_R2.las"), "Synth-1", z2,
              {"GR": a2["GR"], "HLLD": a2["RD"], "RHOZ": a2["RHO"], "TNPH": a2["NPH"], "TENS": np.full(z2.size, 2500.0)},
              {"GR": "GAPI"}, srvc="Schlumberger")
    zl = np.arange(1000.0, 1600.0, 0.1524)
    a3 = logs(zl)
    d = os.path.join(root, "wells", "Synth-LWD", "las")
    os.makedirs(d)
    write_las(os.path.join(d, "lwd_data.las"), "Synth-LWD", zl,
              {"ROP5_RM": np.full(zl.size, 20.0), "GR_ARC": a3["GR"], "P40H": a3["RD"], "P16H": a3["RS"], "A40H": a3["RD"],
               "RHOB": a3["RHO"], "TNPH": a3["NPH"], "PEF": a3["PE"]},
              descs={"GR_ARC": "ARC Gamma Ray", "P40H": "ARC Phase-Shift Resistivity 40-in. at 2 MHz"},
              srvc="Schlumberger D&M")
    d = os.path.join(root, "wells", "Synth-BH", "las")
    os.makedirs(d)
    write_las(os.path.join(d, "HDIL-ZDL-CN.las"), "Synth-BH", zl,
              {"GR": a3["GR"], "M2R9": a3["RD"], "M2R1": a3["RS"], "ZDEN": a3["RHO"], "CNC": a3["NPH"], "MYRES": a3["RD"],
               "CS": np.full(zl.size, 1800.0)}, srvc="Baker Atlas")
    with open(os.path.join(root, "wells_header.csv"), "w") as fh:
        fh.write("Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m\n"
                 "Synth-1,1,-39.6,173.3,25,KB,1700\nSynth-LWD,2,-39.61,173.31,25,KB,1600\nSynth-BH,3,-39.62,173.32,25,KB,1600\n")
    p = Project.from_folder(root)
    p.save(os.path.join(root, "petro.wcproj.json"))
    p = Project.open(os.path.join(root, "petro.wcproj.json"))
    # user tool table: a custom mnemonic
    from .store import petro_dir
    os.makedirs(petro_dir(p), exist_ok=True)
    with open(os.path.join(petro_dir(p), "tool_table.json"), "w") as fh:
        json.dump({"MYTOOL": {"vendor": "Baker Hughes", "desc": "custom", "kind": "induction", "conveyance": "wireline",
                              "families": ["RDEEP"], "mnemonics": ["MYRES"], "res": 0.9}}, fh)
    p.extra_aliases = {"RDEEP": ["MYRES"]}

    def ident(well):
        r = T.identify_well(p, well)
        return {c["mnemonic"]: c for c in r["curves"]}, r

    c1, r1 = ident("Synth-1")
    check(c1["HLLD"]["tool"] == "HALS" and c1["HLLD"]["kind"] == "laterolog" and c1["HLLD"]["vendor"] == "SLB"
          and c1["HLLD"]["conveyance"] == "wireline", f"HLLD -> HALS laterolog SLB wireline ({c1['HLLD']['tool']}, "
          f"{c1['HLLD']['kind']}, {c1['HLLD']['conveyance']})")
    check(c1["DTCO"]["tool"] == "DSI" and abs(c1["DTCO"]["eff_res_m"] - 1.07) < 1e-6, "DTCO -> DSI, resolution 1.07 m")
    check(c1["TNPH"]["tool"] == "CNL" and c1["TNPH"]["notes"].get("neutron_matrix") == "limestone",
          "TNPH in a wireline file -> CNL (limestone matrix note)")
    check(abs(c1["PEFZ"]["eff_res_m"] - 2 * 0.1524) < 1e-3 and c1["PEFZ"]["undersampled"],
          f"PEFZ effective resolution = 2 x sample ({c1['PEFZ']['eff_res_m']:.3f} m)")
    check(r1["files"][0]["services"][:3] == ["PEX", "HALS", "DSI"], f"service tokens from the file name {r1['files'][0]['services'][:4]}")
    c2, r2 = ident("Synth-LWD")
    check(r2["files"][0]["conveyance"] == "LWD" and c2["P40H"]["tool"] == "ARC" and c2["P40H"]["kind"] == "propagation",
          f"LWD file -> ARC propagation ({r2['files'][0]['conveyance_reason']})")
    check(c2["TNPH"]["tool"] == "ADN" and c2["TNPH"]["kind"] == "neutron" and c2["A40H"]["tool_res_m"] == 0.9,
          "TNPH in an LWD file -> ADN neutron; ARC attenuation resolution per mnemonic")
    c3, r3 = ident("Synth-BH")
    check(c3["M2R9"]["tool"] == "HDIL" and c3["M2R9"]["kind"] == "induction" and c3["M2R9"]["vendor"] == "Baker Hughes"
          and c3["CNC"]["tool"] == "CN", "Baker: M2R9 -> HDIL induction, CNC -> CN")
    check(c3["MYRES"]["tool"] == "MYTOOL" and c3["MYRES"]["tool_res_m"] == 0.9, "user tool_table.json entry used")
    tp = os.path.join(tmp, "tools.json")
    T.write_tool_table(tp)
    check("HALS" in T.tool_table(path=tp), "tool table JSON written + read back")

    rep = Q.well_qc(p, "Synth-1")
    shifts = [s for s in rep["shifts"] if s["file"].startswith("PEX-HALS_R2")]
    check(shifts and abs(shifts[0]["shift_m"] + 1.37) < 0.1, f"well QC: run 2 depth shift {shifts[0]['shift_m'] if shifts else '-'} m")
    check(any("depth shift" in w for w in rep["warnings"]), "depth-shift warning raised")
    fam = rep["families"]
    check(abs(fam["CALI"]["bad_hole_m"] - 10.0) < 0.5, f"bad hole 10 m from HCAL vs BS ({fam['CALI']['bad_hole_m']} m)")
    check(fam["DT"]["spikes"] >= 3 and fam["DT"]["tools"] == ["DSI"], f"DT spikes {fam['DT']['spikes']} (3 injected)")
    check(fam["RDEEP"]["kind"] == "laterolog" and fam["RDEEP"]["conveyance"] == "wireline", "family table: kind + conveyance")
    beds = rep["beds"]
    check(beds["from"] == "PE" and beds["thin"]["DT"]["thin_beds"] > beds["thin"]["RHOB"]["thin_beds"],
          f"thin beds: more below DT resolution ({beds['thin']['DT']['thin_beds']}) than RHOB ({beds['thin']['RHOB']['thin_beds']})")
    arr = rep["_qc_arrays"]
    w = p.load_well("Synth-1")
    k = np.searchsorted(w.depth, 1250.15)
    check(arr["QC_THIN_DT"][k] == 1.0 and arr["QC_THIN_PE"][k] == 0.0,
          "0.3 m sand at 1250 m: thin for DT (1.07 m), resolved by PE")
    from .store import load_curves
    dz, saved = load_curves(p, "Synth-1")
    check(dz is not None and "QC_BADHOLE" in saved and len(dz) == len(w.depth), "QC curves saved in <project>_petro/Synth-1.npz")
    check(os.path.exists(os.path.join(petro_dir(p), "qc", "Synth-1.json")), "QC report JSON saved")
    allr = Q.qc_project(p)
    with open(os.path.join(petro_dir(p), "qc_summary.csv")) as fh:
        rows = fh.read().splitlines()
    check(len(allr) == 3 and len(rows) > 10 and rows[0].startswith("Well,Family"), f"qc_summary.csv ({len(rows) - 1} rows)")
    txt = Q.format_report(allr["Synth-LWD"])
    check("ARC" in txt and "LWD" in txt, "text report")

    # ---------------- real data (when present)
    here = os.path.dirname(os.path.abspath(__file__))
    nz = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "data", "nz", "NZ Well Correlation.wcproj.json"))
    if os.path.exists(nz):
        pr = Project.open(nz)
        pr.petro_out = os.path.join(tmp, "nz_petro")
        if "Tui-1" in pr.wells:
            c, r = T.identify_well(pr, "Tui-1")["curves"], None
            hl = [x for x in c if x["mnemonic"] == "HLLD"]
            check(hl and hl[0]["tool"] == "HALS" and hl[0]["how"].startswith("DLIS tool") and hl[0]["vendor"] == "SLB"
                  and hl[0]["conveyance"] == "wireline", "Tui-1 DLIS: HLLD -> HALS from the TOOL object, SLB wireline")
            rep = Q.well_qc(pr, "Tui-1")
            check(rep["families"]["RHOB"]["tools"] == ["TLD"], "Tui-1 RHOB from TLD (Platform Express)")
        if "Rahi-1" in pr.wells:
            rep = Q.well_qc(pr, "Rahi-1", save=False)
            check(rep.get("barite_suspected") == "likely", "Rahi-1: barite mud flagged (PE ~6.7 in clean sand)")
        if "Kiwi-1" in pr.wells:
            rep = Q.well_qc(pr, "Kiwi-1")
            check(rep["files"][0]["conveyance"] == "LWD" and "ARC" in rep["families"]["RDEEP"]["tools"], "Kiwi-1 = SLB LWD (ARC)")
    else:
        print("SKIP real-data petro QC (data/nz not found)")


def run_interp(check, tmp):
    """20b: equations against hand values / round trips, a forward-modelled well, parameter sets, real data."""
    from ..project import Project
    from . import interp as I
    from .store import petro_dir, load_curves
    rng = np.random.default_rng(3)
    # ---- equations
    check(abs(float(I.sw_archie(20.0, 0.05, 0.2)) - 0.25) < 1e-12, "Archie Sw = 0.25 (hand value)")
    sim = float(I.sw_simandoux(10.0, 0.05, 0.2, 0.2, 2.0))
    hand = 0.05 / (2 * 0.04) * (np.sqrt(0.1 ** 2 + 4 * 0.04 / (0.05 * 10)) - 0.1)
    check(abs(sim - hand) < 1e-12 and abs(hand - 0.29653) < 1e-4, f"Simandoux Sw = {sim:.5f} (hand 0.29653)")
    i = np.array([0.5])
    vals = {mth: float(I.vsh_from_igr(i, mth)[0]) for mth in ("larionov_tertiary", "larionov_old", "clavier", "steiber")}
    want = {"larionov_tertiary": 0.083 * (2 ** 1.85 - 1), "larionov_old": 0.33 * (2 ** 1.0 - 1),
            "clavier": 1.7 - np.sqrt(3.38 - 1.2 ** 2), "steiber": 0.25}
    check(all(abs(vals[k] - want[k]) < 1e-12 for k in want), "Vsh: Larionov (2), Clavier, Steiber at IGR 0.5")
    phi = rng.uniform(0.08, 0.3, 200)
    vsh = rng.uniform(0.0, 0.4, 200)
    sw0 = rng.uniform(0.15, 1.0, 200)
    rw, rsh, m, n = 0.05, 2.5, 2.0, 2.2
    ct = phi ** m * sw0 ** n / (rw * (1 - vsh)) + vsh * sw0 / rsh
    sw = I.sw_modified_simandoux(1 / ct, rw, phi, vsh, rsh, 1.0, m, n)
    check(np.max(np.abs(sw - sw0)) < 1e-6, "modified Simandoux: forward -> inverse round trip")
    rt = 1.0 / ((vsh ** (1 - vsh / 2) / np.sqrt(rsh) + phi ** (m / 2) / np.sqrt(rw)) * sw0 ** (n / 2)) ** 2
    check(np.max(np.abs(I.sw_indonesia(rt, rw, phi, vsh, rsh, 1.0, m, n) - sw0)) < 1e-9, "Indonesia: round trip")
    qv, T = 0.4 * vsh, 80.0
    B = I.juhasz_b(T, rw)
    ct = phi ** m * sw0 ** n * (1 / rw + B * qv / sw0)
    check(np.max(np.abs(I.sw_waxman_smits(1 / ct, rw, phi, qv, T, 1.0, m, n) - sw0)) < 1e-6, "Waxman-Smits: round trip")
    b25 = float(I.juhasz_b(25.0, 0.05))
    check(3.5 < b25 < 4.6 and float(B) > b25, f"Juhasz B {b25:.2f} at 25 degC (Waxman-Smits ~4), rises with T ({float(B):.1f} at 80)")
    swb, rwb = np.minimum(vsh * 0.3, sw0 * 0.9), 0.2
    ct = phi ** m * sw0 ** n * (1 / rw + swb / sw0 * (1 / rwb - 1 / rw))
    check(np.max(np.abs(I.sw_dual_water(1 / ct, rw, phi, swb, rwb, 1.0, m, n) - sw0)) < 1e-6, "dual water: round trip")
    phit, gas = I.phi_nd(np.array([0.30, 0.20]), np.array([0.10, 0.22]))
    check(abs(phit[0] - np.sqrt(0.05)) < 1e-12 and gas[0] and abs(phit[1] - 0.21) < 1e-12 and not gas[1],
          "N-D porosity: gas crossover uses sqrt((PHID^2 + NPHI^2) / 2)")
    check(abs(float(I.perm_timur(0.2, 0.2)) - 0.136 * 20 ** 4.4 / 20 ** 2) < 1e-9 * 200, "Timur = 0.136 phi%^4.4 / Swi%^2")
    check(abs(float(I.perm_coates(0.2, 0.2)) - 256.0) < 1e-9 and abs(float(I.perm_wyllie_rose(0.2, 0.2)) - 100.0) < 1e-9,
          "Coates / Wyllie-Rose hand values")
    toc, dlr = I.toc_passey(50.0, 100.0, 5.0, 100.0, 10.0)
    check(abs(float(dlr) - 1.0) < 1e-12 and abs(float(toc) - 10 ** (2.297 - 1.688)) < 1e-9, "Passey TOC (dlogR = 1, LOM 10)")
    check(abs(float(I.arps(0.1, 25.0, 100.0)) - 0.1 * 46.5 / 121.5) < 1e-12, "Arps temperature conversion")
    K = 61 + 0.133 * (80 * 1.8 + 32)
    ssp = -K * np.log10(0.85 * 0.5 / 0.05)
    check(abs(I.rw_from_sp(ssp, 0.5, 80.0) - 0.05) < 1e-9, "Rw from SSP (inverse of the SSP equation)")
    # Pickett: 60 % water-bearing points (m = 1.9, Rw = 0.06) + 40 % hydrocarbon points above the line
    phi = rng.uniform(0.06, 0.3, 600)
    swp = np.where(rng.uniform(size=600) < 0.6, 1.0, rng.uniform(0.2, 0.6, 600))
    rt = 0.06 / (phi ** 1.9 * swp ** 2) * np.exp(0.03 * rng.standard_normal(600))
    fit = I.pickett_fit(rt, phi, m=None)
    check(abs(fit["m"] - 1.9) < 0.1 and abs(fit["rw"] / 0.06 - 1) < 0.08, f"Pickett fit m {fit['m']:.2f} (1.9), Rw {fit['rw']:.4f} (0.06)")
    fit2 = I.pickett_fit(rt, phi, m=1.9)
    check(abs(fit2["rw"] / 0.06 - 1) < 0.05, f"Pickett with fixed m: Rw {fit2['rw']:.4f}")

    # ---- forward-modelled well
    root = os.path.join(tmp, "interp_proj")
    d = os.path.join(root, "wells", "Fwd-1", "las")
    os.makedirs(d)
    z = np.arange(950.0, 1260.0, 0.1524)
    V = np.full(z.size, 0.9)
    P = np.full(z.size, 0.0)
    S = np.ones(z.size)
    pay = (z >= 1000) & (z < 1040)
    wat = (z >= 1045) & (z < 1095)
    shaly = (z >= 1110) & (z < 1190)
    V[pay | wat], P[pay | wat], S[pay] = 0.1, 0.22, 0.3
    V[shaly], P[shaly] = 0.5, 0.05
    rho_ma, rho_sh, nphi_sh, rw, rsh = 2.65, 2.55, 0.35, 0.05, 2.0
    rhob = (1 - P - V) * rho_ma + V * rho_sh + P * 1.0
    nphi = P + V * nphi_sh
    gr = 20 + 100 * V
    ct = P ** 2 * S ** 2 / rw + V * S / rsh
    rt = 1 / ct
    dt = 55.5 + (P + 0.2 * V) * (189 - 55.5)
    e = lambda s_: s_ * rng.standard_normal(z.size)  # noqa: E731  tiny noise: constant stretches would be blanked as flat lines
    gr, rhob, nphi, rt, dt = gr + e(0.05), rhob + e(0.0003), nphi + e(0.0003), rt * np.exp(e(0.002)), dt + e(0.05)
    write_las(os.path.join(d, "fwd.las"), "Fwd-1", z, {"GR": gr, "RHOB": rhob, "NPHI": nphi, "LLD": rt, "LLS": rt, "DT": dt},
              {"NPHI": "V/V"}, srvc="Schlumberger")
    with open(os.path.join(root, "wells_header.csv"), "w") as fh:
        fh.write("Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m\nFwd-1,1,-39.6,173.3,0,KB,1260\n")
    p = Project.from_folder(root)
    p.formations = [{"name": "A", "color": "#ff0000"}, {"name": "B", "color": "#00ff00"}, {"name": "C", "color": "#0000ff"}]
    p.tops = {"Fwd-1": {"A": 1000.0, "B": 1100.0, "C": 1200.0}}
    p.save(os.path.join(root, "fwd.wcproj.json"))
    p = Project.open(os.path.join(root, "fwd.wcproj.json"))
    pset = {"default": {"gr_clean": 20.0, "gr_shale": 120.0, "rho_sh": rho_sh, "nphi_sh": nphi_sh, "rw": rw, "rsh": rsh,
                        "sw_method": "simandoux", "vsh_cut": 0.4, "phi_cut": 0.08, "sw_cut": 0.6}}
    I.save_params(p, pset)
    r = I.interpret_well(p, "Fwd-1")
    c = r["curves"]
    kz = (r["depth"] >= 1002) & (r["depth"] < 1038)
    check(np.nanmax(np.abs(c["PHIE"][kz] - 0.22)) < 0.005 and np.nanmax(np.abs(c["VSH"][kz] - 0.1)) < 0.005,
          f"forward well: PHIE {np.nanmedian(c['PHIE'][kz]):.3f} (0.22), Vsh {np.nanmedian(c['VSH'][kz]):.3f} (0.10)")
    check(np.nanmax(np.abs(c["SW"][kz] - 0.3)) < 0.01, f"forward well: Simandoux Sw {np.nanmedian(c['SW'][kz]):.3f} (0.30)")
    row = {q["Zone"]: q for q in r["zones"]}
    check(abs(row["A"]["NetPay_TVD"] - 40.0) < 0.5 and abs(row["A"]["NetRes_TVD"] - 90.0) < 0.8,
          f"zone A net pay {row['A']['NetPay_TVD']} m (40), net res {row['A']['NetRes_TVD']} m (90)")
    check(abs(row["A"]["SoPhiH"] - 40 * 0.22 * 0.7) < 0.15 and row["B"]["NetPay_TVD"] == 0.0,
          f"SoPhiH {row['A']['SoPhiH']} ({40 * 0.22 * 0.7:.2f}); shaly zone B no pay")
    check(row["A"]["Sw_pay"] is not None and abs(row["A"]["Sw_pay"] - 0.3) < 0.01 and row["A"]["kh_pay"] > 0, "pay Sw + kh")
    sr = I.read_summary(p)
    check(len(sr) == 3 and sr[0]["Well"] == "Fwd-1", "summary.csv written (one row per zone)")
    dz, saved = load_curves(p, "Fwd-1")
    check("PHIE" in saved and "SW" in saved and "NET_PAY" in saved, "computed curves saved to the well's npz")
    # Archie overestimates Sw-hydrocarbon (lower Sw) in shaly sand? (shale conductivity ignored -> higher Sw)
    pa = I.copy_params(pset)
    pa["default"]["sw_method"] = "archie"
    ra = I.interpret_well(p, "Fwd-1", pa, save=False)
    check(np.nanmedian(ra["curves"]["SW"][kz]) > np.nanmedian(c["SW"][kz]) + 0.01,
          "Archie Sw higher than Simandoux in shaly sand (shale conductance ignored)")
    # Rw from Rwa minimum (water leg in zone A): Rwa = phi^2 Rt in the water leg ~ Rw (shale term small)
    pb = I.copy_params(pset)
    pb["default"].pop("rw")
    rb = I.interpret_well(p, "Fwd-1", pb, save=False)
    check(abs(rb["used"]["A"]["rw"] / rw - 1) < 0.35 and "Rwa" in rb["used"]["A"]["rw_how"],
          f"Rw auto from the Rwa minimum: {rb['used']['A']['rw']:.4f} (true 0.05; shale conductance biases it low)")
    # parameter precedence
    ps = {"default": {"m": 2.1}, "zones": {"A": {"m": 1.8}}, "wells": {"Fwd-1": {"A": {"m": 1.7}, "*": {"n": 2.3}}}}
    check(I.resolve(ps, "Fwd-1", "A")["m"] == 1.7 and I.resolve(ps, "X", "A")["m"] == 1.8 and I.resolve(ps, "X", "B")["m"] == 2.1
          and I.resolve(ps, "Fwd-1", "B")["n"] == 2.3 and I.resolve(ps, "X", "B")["n"] == 2.0, "parameter sets: well+zone > zone > default")
    sens = I.cutoff_sensitivity(r, "A")
    npay = sens["phi_cut"]["net_pay_tvd"]
    check(np.all(np.diff(npay) <= 1e-9) and npay[0] >= 40 - 0.5 and npay[-1] < 1e-9,
          "cutoff sensitivity: net pay falls monotonically with the porosity cut-off")
    swc = sens["sw_cut"]["net_pay_tvd"]
    check(np.all(np.diff(swc) >= -1e-9) and swc[-1] > 85, "cutoff sensitivity: Sw cut 1.0 makes all net reservoir pay")
    pk = I.pickett_data(r, "A")
    check(len(pk["phi"]) > 100 and pk["rw"] == rw, "Pickett plot data")
    # TOC on a source-rock interval
    pt = I.copy_params(pset)
    pt["zones"] = {"B": {"toc": True, "rt_base": 2.0, "dt_base": 100.0}}
    rt_ = I.interpret_well(p, "Fwd-1", pt, save=False)
    check(np.isfinite(rt_["curves"]["TOC"]).any() and not np.isfinite(rt_["curves"]["TOC"][kz]).any(), "TOC only where asked (zone B)")

    # ---- real data (Maui-5 / Tui-1, default parameters)
    here = os.path.dirname(os.path.abspath(__file__))
    nz = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "data", "nz", "NZ Well Correlation.wcproj.json"))
    if os.path.exists(nz):
        pr = Project.open(nz)
        pr.petro_out = os.path.join(tmp, "nz_petro")
        for wn in ("Maui-5", "Tui-1"):
            if wn not in pr.wells or not pr.tops.get(wn):
                continue
            rr = I.interpret_well(pr, wn)
            ok = [q for q in rr["zones"] if q["PHIE_res"] is not None]
            good = all(0.03 < q["PHIE_res"] < 0.35 for q in ok) and all(np.isfinite(q["SoPhiH"]) for q in rr["zones"])
            check(ok and good, f"{wn} real run: " + "; ".join(f"{q['Zone'][:12]} net pay {q['NetPay_TVD']} m PHIE "
                                                           f"{q['PHIE_res']} Sw {q['Sw_pay']}" for q in rr["zones"]))
    else:
        print("SKIP real-data petro interpretation (data/nz not found)")
    return p


def run_multimin(check, tmp, fwd_project=None):
    """20c: constrained solver (analytic, bounds vs brute force), synthetic mineral well, Monte Carlo, real data."""
    import time
    from ..project import Project
    from . import interp as I
    from . import multimin as M
    rng = np.random.default_rng(11)
    comps = ["quartz", "calcite", "water"]
    E = M.endpoint_matrix(comps)
    v0 = np.array([0.6, 0.2, 0.2])
    d = E @ v0
    V, mis, rec = M.solve_volumes(d[None, :], E, [M.SIGMA[k] for k in M.LOGS])
    check(np.max(np.abs(V[0] - v0)) < 1e-8 and mis[0] < 1e-12, "multimin: exact 2-mineral + water case recovered")
    # bound active: data from quartz + water but RHOB / U pulled below quartz -> calcite would go negative
    d2 = M.endpoint_matrix(["quartz", "water"]) @ np.array([0.8, 0.2])
    d2 = d2 + np.array([0, -0.06, 0, 0, -0.5, 0])
    V2, _, _ = M.solve_volumes(d2[None, :], E, [M.SIGMA[k] for k in M.LOGS])
    w = 1 / np.array([M.SIGMA[k] for k in M.LOGS])
    g = np.arange(0, 1.0001, 0.002)
    A, B = np.meshgrid(g, g, indexing="ij")
    C = 1 - A - B
    ok = C >= -1e-12
    X = np.stack([A[ok], B[ok], C[ok]], 1)
    obj = (((X @ E.T - d2) * w) ** 2).sum(1)
    xb = X[np.argmin(obj)]
    check(V2[0, 1] == 0.0 and np.max(np.abs(V2[0] - xb)) < 0.004, f"multimin: non-negativity active (calcite 0), matches brute "
          f"force {np.round(V2[0], 3)} vs {np.round(xb, 3)}")
    # missing logs per sample (two availability patterns)
    D = np.vstack([d, d])
    D[1, 3] = np.nan
    V3, _, _ = M.solve_volumes(D, E, [M.SIGMA[k] for k in M.LOGS])
    check(np.max(np.abs(V3 - v0)) < 1e-8, "multimin: rows with a missing log solved with the remaining logs")
    # speed: 20,000 samples, 6 unknowns
    comps6 = ["quartz", "calcite", "dolomite", "illite", "water", "gas"]
    E6 = M.endpoint_matrix(comps6)
    Vt = rng.dirichlet(np.ones(6), 20000)
    Dt = Vt @ E6.T
    t0 = time.perf_counter()
    V6, _, _ = M.solve_volumes(Dt, E6, [M.SIGMA[k] for k in M.LOGS])
    dt_s = time.perf_counter() - t0
    check(dt_s < 10.0 and np.nanmax(np.abs(V6 - Vt)) < 1e-6, f"multimin: 20,000 samples x 6 components in {dt_s:.2f} s, exact")

    # ---- synthetic well with known mineralogy (quartz + illite + water + gas)
    root = os.path.join(tmp, "mm_proj")
    dd = os.path.join(root, "wells", "Min-1", "las")
    os.makedirs(dd)
    z = np.arange(1500.0, 1700.0, 0.1524)
    nn = z.size
    vill = np.clip(0.15 + 0.1 * np.sin(z / 3.0) + 0.02 * rng.standard_normal(nn), 0.02, 0.5)
    phi = np.clip(0.2 - 0.25 * (vill - 0.15) + 0.01 * rng.standard_normal(nn), 0.05, 0.3)
    sw = np.where(z < 1600, 0.35, 1.0)
    vq = 1 - vill - phi
    Vtrue = np.column_stack([vq, vill, phi * sw, phi * (1 - sw)])
    Em = M.endpoint_matrix(["quartz", "illite", "water", "gas"])
    logs = Vtrue @ Em.T
    noise = np.array([M.SIGMA[k] for k in M.LOGS]) * 0.3
    logs = logs + noise * rng.standard_normal(logs.shape)
    rw = 0.05
    rt = rw / np.maximum(logs[:, 5], 1e-3) ** 2
    pef = logs[:, 4] / logs[:, 1]
    write_las(os.path.join(dd, "min.las"), "Min-1", z, {"GR": logs[:, 0], "RHOB": logs[:, 1], "NPHI": logs[:, 2],
              "DT": logs[:, 3], "PEF": pef, "LLD": rt}, {"NPHI": "V/V"}, srvc="Schlumberger")
    with open(os.path.join(root, "wells_header.csv"), "w") as fh:
        fh.write("Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m\nMin-1,1,-39.6,173.3,0,KB,1700\n")
    p = Project.from_folder(root)
    p.formations = [{"name": "R", "color": "#ffff00"}, {"name": "Base", "color": "#000000"}]
    p.tops = {"Min-1": {"R": 1510.0, "Base": 1690.0}}
    p.save(os.path.join(root, "mm.wcproj.json"))
    p = Project.open(os.path.join(root, "mm.wcproj.json"))
    pset = {"default": {"rw": rw, "mm_minerals": ["quartz", "illite"], "mm_fluid": "gas", "phi_cut": 0.08,
                        "sw_cut": 0.6, "vsh_cut": 0.4}}
    r = M.multimin_well(p, "Min-1", pset=pset)
    zz = r["depth"]
    sel = (zz > 1512) & (zz < 1688)
    tv = lambda a: np.interp(zz[sel], z, a)  # noqa: E731
    e_phi = np.nanmedian(np.abs(r["curves"]["MM_PHIE"][sel] - tv(phi)))
    e_ill = np.nanmedian(np.abs(r["volumes"]["illite"][sel] - tv(vill)))
    gsel = sel & (zz < 1595)
    e_sw = np.nanmedian(np.abs(r["curves"]["MM_SW"][gsel] - 0.35))
    check(e_phi < 0.02 and e_ill < 0.03 and e_sw < 0.06, f"synthetic mineral well: median errors PHIE {e_phi:.3f}, illite "
          f"{e_ill:.3f}, gas-leg Sw {e_sw:.3f}")
    mis = np.nanmedian(r["curves"]["MM_MISFIT"][sel])
    check(mis < 1.0, f"reconstructed logs fit within the log uncertainties (median chi2/eq {mis:.2f})")
    check(np.nanmedian(np.abs(r["curves"]["MM_REC_RHOB"][sel] - np.interp(zz[sel], z, logs[:, 1]))) < 0.02,
          "reconstructed RHOB matches the input")
    row = r["zones"][0]
    true_pay = 1600 - 1510
    check(abs(row["NetPay_TVD"] - true_pay) < 3.0 and row["V_illite"] is not None,
          f"multimin zone: net pay {row['NetPay_TVD']} m (gas leg {true_pay} m), mean illite {row['V_illite']}")
    from .store import load_curves
    _, saved = load_curves(p, "Min-1")
    check("MM_V_ILLITE" in saved and "MM_PHIE" in saved, "multimin volumes saved (MM_ curves)")

    # ---- Monte Carlo on the deterministic chain (forward well from run_interp)
    if fwd_project is not None:
        ps = I.load_params(fwd_project)
        mc0 = M.monte_carlo(fwd_project, "Fwd-1", dists={"m": ("normal", 0.0)}, n=5, pset=ps)
        a0 = mc0["zones"]["A"]["SoPhiH"]
        check(abs(a0["P90"] - a0["P10"]) < 1e-9, "Monte Carlo with zero spread = deterministic answer")
        t0 = time.perf_counter()
        mc = M.monte_carlo(fwd_project, "Fwd-1", n=100, pset=ps, seed=4)
        t_mc = time.perf_counter() - t0
        a = mc["zones"]["A"]["SoPhiH"]
        true = 40 * 0.22 * 0.7
        check(a["P90"] < a["P50"] < a["P10"] and a["P90"] <= true <= a["P10"],
              f"Monte Carlo SoPhiH P90 {a['P90']:.2f} < P50 {a['P50']:.2f} < P10 {a['P10']:.2f} brackets the truth "
              f"{true:.2f} ({t_mc:.1f} s for 100)")
        mcw = M.monte_carlo(fwd_project, "Fwd-1", dists={"rw_mult": ("lognormal", 0.2)}, n=100, pset=ps, seed=4)
        mcn = M.monte_carlo(fwd_project, "Fwd-1", dists={"rw_mult": ("lognormal", 0.05)}, n=100, pset=ps, seed=4)
        sw_ = mcw["zones"]["A"]["SoPhiH"]
        sn_ = mcn["zones"]["A"]["SoPhiH"]
        check(sw_["P10"] - sw_["P90"] > sn_["P10"] - sn_["P90"], "wider Rw uncertainty -> wider SoPhiH range")
        rows = M.mc_table(mc)
        check(any(rr["Zone"] == "A" and rr["SoPhiH_P50"] is not None for rr in rows), "Monte-Carlo table rows")
        mcm = M.monte_carlo(p, "Min-1", n=20, pset=pset, method="multimin", seed=2)
        s = mcm["zones"]["R"]["PHIH_pay"]
        check(s["P90"] <= s["P50"] <= s["P10"] and s["P10"] > s["P90"], f"Monte Carlo on the mineral solver (end-points "
              f"perturbed): PHIH P90 {s['P90']:.2f} / P50 {s['P50']:.2f} / P10 {s['P10']:.2f}")

    # ---- real data
    here = os.path.dirname(os.path.abspath(__file__))
    nz = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "data", "nz", "NZ Well Correlation.wcproj.json"))
    if os.path.exists(nz):
        pr = Project.open(nz)
        pr.petro_out = os.path.join(tmp, "nz_petro")
        if "Maui-7" in pr.wells and pr.tops.get("Maui-7"):
            ps = {"default": {"mm_minerals": ["quartz", "illite", "calcite"], "mm_fluid": "gas"}}
            det = I.interpret_well(pr, "Maui-7", ps, save=False)
            rr = M.multimin_well(pr, "Maui-7", pset=ps, det=det, save=False)
            zc = [q for q in rr["zones"] if q["Zone"].startswith("C-sands")]
            dc = [q for q in det["zones"] if q["Zone"].startswith("C-sands")]
            if zc and dc:
                check(zc[0]["PHIE_res"] is not None and 0.05 < zc[0]["PHIE_res"] < 0.3 and zc[0]["Misfit_med"] is not None,
                      f"Maui-7 C-sands multimin (qtz+illite+calcite+gas): PHIE {zc[0]['PHIE_res']} vs deterministic "
                      f"{dc[0]['PHIE_res']}, net pay {zc[0]['NetPay_TVD']} vs {dc[0]['NetPay_TVD']} m, misfit "
                      f"{zc[0]['Misfit_med']}, mean V quartz {zc[0].get('V_quartz')} illite {zc[0].get('V_illite')}")
    else:
        print("SKIP real-data multimin (data/nz not found)")


def run_rockphysics(check, tmp, fwd_project=None):
    """20d: Batzle-Wang fluids, Gassmann round trips, Greenberg-Castagna, fizz gas, elastic identities, well run."""
    from ..project import Project
    from . import rockphysics as R
    rb, vb, kb = R.brine(20.0, 0.1, 0.0)
    check(abs(float(rb) - 0.998) < 0.003 and abs(float(vb) - 1482.3) < 2.0, f"Batzle-Wang pure water 20 degC: "
          f"{float(rb):.4f} g/cc, {float(vb):.1f} m/s (0.998, 1482)")
    rb2, vb2, _ = R.brine(80.0, 30.0, 100000.0)
    check(1.03 < float(rb2) < 1.09 and 1550 < float(vb2) < 1800, f"brine 100k ppm 80 degC 30 MPa: {float(rb2):.3f} g/cc, "
          f"{float(vb2):.0f} m/s")
    rg, kg = R.gas(100.0, 30.0, 0.6)
    check(0.15 < float(rg) < 0.25 and 0.04 < float(kg) < 0.2, f"Batzle-Wang gas G 0.6, 100 degC, 30 MPa: {float(rg):.3f} g/cc, "
          f"K {float(kg) * 1000:.0f} MPa")
    ro, vo, ko = R.oil(15.6, 0.1, 35.0, 0.0)
    check(abs(float(ro) - 141.5 / 166.5) < 0.012 and 1300 < float(vo) < 1600, f"dead oil API 35 at surface: {float(ro):.3f} g/cc "
          f"(0.850), {float(vo):.0f} m/s")
    ro2, _, _ = R.oil(80.0, 30.0, 35.0, 100.0)
    check(float(ro2) < float(ro), "live oil at reservoir conditions lighter than dead oil at surface")
    km = 37.0
    ks = np.array([12.0, 18.0, 25.0])
    phi = np.array([0.3, 0.2, 0.1])
    kd = R.gassmann_dry(ks, km, 2.5, phi)
    check(np.allclose(R.gassmann_sat(kd, km, 2.5, phi), ks, rtol=1e-12) and np.all(kd < ks), "Gassmann sat -> dry -> sat round trip")
    vp, vs, rho = np.array([3000.0, 3500.0]), np.array([1500.0, 1900.0]), np.array([2.2, 2.35])
    ph = np.array([0.25, 0.18])
    kb_, rb_ = 2.6, 1.03
    kh_, rh_ = R.fluid_mix(0.2, 2.6, 0.08, 1.03, 0.2)
    a = R.substitute(vp, vs, rho, ph, km, kb_, rb_, kh_, rh_)
    b = R.substitute(a[0], a[1], a[2], ph, km, kh_, rh_, kb_, rb_)
    check(np.allclose(b[0], vp) and np.allclose(b[1], vs) and np.allclose(b[2], rho) and np.all(a[0] < vp) and np.all(a[1] > vs),
          "fluid substitution brine -> gas -> brine round trip (gas: Vp down, Vs up)")
    v1 = R.vs_greenberg_castagna(np.array([3000.0]), {"sandstone": np.array([1.0])})
    check(abs(float(v1[0]) - (0.80416 * 3 - 0.85588) * 1000) < 1e-6, "Greenberg-Castagna sandstone line")
    v2 = R.vs_greenberg_castagna(np.array([3000.0]), {"sandstone": np.array([0.5]), "shale": np.array([0.5])})
    s1, s2 = 0.80416 * 3 - 0.85588, 0.76969 * 3 - 0.86735
    hand = 500 * (0.5 * s1 + 0.5 * s2 + 1 / (0.5 / s1 + 0.5 / s2))
    check(abs(float(v2[0]) - hand) < 1e-6, "Greenberg-Castagna lithology mix (arithmetic + harmonic average)")
    # fizz gas: 7 % gas saturation drops Vp almost as much as 80 % gas, density barely
    kf, rf = R.fluid_mix(0.93, 2.6, 0.08, 1.03, 0.2)
    kg8, rg8 = R.fluid_mix(0.2, 2.6, 0.08, 1.03, 0.2)
    fz = R.substitute(vp, vs, rho, ph, km, kb_, rb_, kf, rf)
    g8 = R.substitute(vp, vs, rho, ph, km, kb_, rb_, kg8, rg8)
    dvp_f, dvp_g = vp - fz[0], vp - g8[0]
    drho_f, drho_g = rho - fz[2], rho - g8[2]
    check(np.all(dvp_f > 0.7 * dvp_g) and np.all(drho_f < 0.15 * drho_g),
          f"fizz gas: Vp drop {100 * dvp_f[0] / dvp_g[0]:.0f}% of commercial gas, density drop {100 * drho_f[0] / drho_g[0]:.0f}% "
          "(only density separates them)")
    e = R.elastic(np.array([3000.0 * np.sqrt(3)]), np.array([3000.0]), np.array([2.4]))
    K, mu = R.moduli(3000.0 * np.sqrt(3), 3000.0, 2.4)
    check(abs(e["PR"][0] - 0.25) < 1e-12 and abs(e["YM"][0] - 9 * K * mu / (3 * K + mu)) < 1e-9
          and abs(e["LR"][0] - 2.4 * (K - 2 / 3 * mu)) < 1e-9 and abs(e["MR"][0] - 2.4 * mu) < 1e-9,
          "elastic identities: PR 0.25 at Vp/Vs sqrt(3); E = 9K mu / (3K + mu); lambda-rho, mu-rho")
    e2 = R.elastic(np.array([4500.0, 3000.0]), np.array([2900.0, 1300.0]), np.array([2.6, 2.3]))
    check(e2["BI"][0] > e2["BI"][1], "brittleness: stiff low-PR rock more brittle")
    if fwd_project is not None:
        from .store import load_curves
        r = R.rock_physics_well(fwd_project, "Fwd-1")
        c = r["curves"]
        zz = r["depth"]
        pay = (zz > 1002) & (zz < 1038)
        sh = (zz > 960) & (zz < 990)
        check(np.isfinite(c["VS"][pay]).all() and "Greenberg" in r["vs_source"][0], f"Vs predicted ({r['vs_source'][0]})")
        check(np.nanmean(c["AI_BRINE"][pay]) > np.nanmean(c["AI_GAS"][pay]) and np.nanmean(c["VPVS_GAS"][pay]) < np.nanmean(c["VPVS_BRINE"][pay]),
              f"forward well scenarios: AI brine {np.nanmean(c['AI_BRINE'][pay]):.0f} > gas {np.nanmean(c['AI_GAS'][pay]):.0f}, "
              f"Vp/Vs gas {np.nanmean(c['VPVS_GAS'][pay]):.2f} < brine {np.nanmean(c['VPVS_BRINE'][pay]):.2f}")
        check(np.allclose(c["AI_BRINE"][sh], c["AI"][sh], equal_nan=True), "shale (not net reservoir) unchanged by substitution")
        check(np.nanmean(c["RHO_FIZZ"][pay]) > np.nanmean(c["RHO_GAS"][pay]) + 0.08, "fizz keeps brine-like density")
        _, saved = load_curves(fwd_project, "Fwd-1")
        check("RP_AI" in saved and "RP_VP_FIZZ" in saved and "PHIE" in saved, "RP_ curves saved next to the petro curves")
        rows = R.scenario_table(r, I_zone_rows(fwd_project, "Fwd-1"))
        check(rows and rows[0]["AI_GAS"] is not None, "scenario table per zone")
    here = os.path.dirname(os.path.abspath(__file__))
    nz = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "data", "nz", "NZ Well Correlation.wcproj.json"))
    if os.path.exists(nz):
        pr = Project.open(nz)
        pr.petro_out = os.path.join(tmp, "nz_petro")
        if "Tui-1" in pr.wells and pr.tops.get("Tui-1"):
            r = R.rock_physics_well(pr, "Tui-1", save=False)
            vv = r["curves"]["VPVS"]
            check(np.isfinite(vv).sum() > 100 and 1.4 < np.nanmedian(vv) < 2.6,
                  f"Tui-1 rock physics: Vs from {r['vs_source']}, median Vp/Vs {np.nanmedian(vv):.2f}, "
                  f"median brittleness {np.nanmedian(r['curves']['BI']):.0f}")
    else:
        print("SKIP real-data rock physics (data/nz not found)")


def I_zone_rows(project, well):
    from . import interp as I
    return I.interpret_well(project, well, save=False)["zones"]
