"""Analytics 13a+: engineering + production data checks on the synthetic data set."""
import os

import numpy as np


def run(check, tmp):
    from .analytics import engineering as E
    from .analytics.synthetic import make
    d = os.path.join(tmp, "eng")
    truth = make(d)
    rows = E.read_csv_rows(os.path.join(d, "completions.csv"))
    mp = E.auto_mapping(rows[0].keys(), "Enverus")
    check(mp.get("api") == "API14" and mp.get("proppant") == "Proppant_LBS" and mp.get("x") == "X",
          "column mapping from the Enverus preset + aliases")
    ct = E.completion_table(rows, mp)
    rows_m = E.read_csv_rows(os.path.join(d, "completions_metric.csv"))
    ctm = E.completion_table(rows_m, preset="Enverus")
    check(ctm.meta["units"].get("fluid") == "m3", "fluid in m3 detected from the header")
    k = int(np.nonzero(truth["metric"])[0][0])
    fpf_m = ctm.cols["fluid_bbl_ft"][0]
    check(abs(fpf_m - truth["fpf"][k]) < 0.05, f"m3 converted to bbl ({fpf_m:.2f} vs {truth['fpf'][k]:.2f} bbl/ft)")
    i0 = 1  # first well of completions.csv
    r0 = ct.ids.index(truth["api"][i0])
    check(abs(ct.cols["proppant_lb_ft"][r0] - truth["ppf"][i0]) < 1.0 and
          abs(ct.cols["stage_spacing_ft"][r0] * ct.cols["stages"][r0] - ct.cols["lateral_ft"][r0]) < 1e-6 and
          abs(ct.cols["lateral_ft"][r0] - truth["lat"][i0]) < 1.0, "completion features (proppant/ft, stage spacing)")
    # merge both completion files, spacing + parent/child
    allc = E.completion_table(rows + [{**r, "TotalFluidPumped_BBL": str(float(r["TotalFluid_M3"]) * 6.28981)}
                                      for r in rows_m], E.auto_mapping(rows[0].keys(), "Enverus"))
    allc = E.spacing_features(allc, radius_ft=1500.0)
    order = [allc.ids.index(a) for a in truth["api"]]
    ch = allc.cols["child_well"][order]
    check(np.array_equal(ch, truth["child"]), f"parent/child flags ({int(ch.sum())} child wells)")
    allc.cols["proppant_lb_ft"][3] = 9000.0
    E.qc_flags(allc)
    check(allc.cols["qc_flag"][3] == 1 and allc.cols["qc_flag"].sum() == 1, "unit-error flag (proppant > 5000 lb/ft)")
    # production
    prow = E.read_csv_rows(os.path.join(d, "production.csv"))
    ser = E.monthly_series(prow)
    s0 = ser[truth["api"][0]]
    pf = E.production_features(ser, {a: truth["lat"][i] for i, a in enumerate(truth["api"])})
    j = pf.ids.index(truth["api"][0])
    boe0 = s0["oil"] + s0["gas"] / 6.0
    check(abs(pf.cols["cum12_boe"][j] - boe0[:12].sum()) < 1e-6 and pf.cols["peak_month"][j] == 1,
          "cum 12-month BOE and peak month")
    check(abs(pf.cols["gor_scf_bbl"][j] - (1500 + 1000 * truth["x"][0] / 20000)) < 5, "GOR")
    short = [a for a in pf.ids if pf.cols["months_on"][pf.ids.index(a)] < 24]
    check(all(np.isnan(pf.cols["cum24_boe"][pf.ids.index(a)]) for a in short), "wells with < 24 months: no 24-month cum")
    jj = [pf.ids.index(a) for a in truth["api"]]
    b_est = pf.cols["arps_b"][jj]
    long_ = pf.cols["months_on"][jj] >= 36
    eur_est = pf.cols["eur_boe"][jj] / 1000.0
    rel = np.abs(eur_est - truth["eur_kboe"])[long_] / truth["eur_kboe"][long_]
    check(np.median(np.abs(b_est[long_] - 1.0)) < 0.1 and np.median(rel) < 0.05,
          f"Arps fit (median b {np.median(b_est[long_]):.2f}, true 1.0) and EUR within {np.median(rel):.1%}")
    check(np.isfinite(pf.cols["cum12_boe_per1000ft"][j]), "per-1000-ft normalised cums")
    # FracFocus + lease allocation
    wbbl, plb = E.fracfocus_estimates(4_200_000, 88.0, 11.0)
    check(abs(wbbl - 100000) < 1 and abs(plb - 4_200_000 * 8.34 / 0.88 * 0.11) < 1, "FracFocus water + proppant estimate")
    al = E.allocate_lease(900.0, ["a", "b", "c"], "test", weights=[1, 2, 6])
    check(abs(al["c"] - 600.0) < 1e-9 and abs(sum(al.values()) - 900) < 1e-9, "lease allocation by well test")
    j2 = E.join_on_api(allc, pf)
    check("cum12_boe" in j2.cols and np.isfinite(j2.cols["cum12_boe"]).mean() > 0.9, "completion + production joined by API")
    return j2, truth
