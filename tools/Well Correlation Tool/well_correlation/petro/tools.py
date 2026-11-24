"""Logging tool / vendor / service identification and vertical resolution (PLAN 20a).

Why: corrections and default parameters depend on the tool (array induction vs laterolog, wireline vs LWD,
limestone-calibrated neutron, ...).  Evidence used per curve, strongest first:

  1. DLIS TOOL objects (GENERIC-NAME / TRADEMARK-NAME + the tool's CHANNELS list) and ORIGIN PRODUCER-NAME
  2. distinctive mnemonics (HLLD = HALS, AT90 = AIT, M2R9 = HDIL, P40H = ARC, ...)
  3. service tokens in the file name (DLL-MSFL-GR-SP-AMS.LAS) and LAS ~Well OS / SRVC lines
  4. curve descriptions ("Laterolog Deep Resistivity", "ARC Phase-Shift Resistivity ...")
  5. a generic tool per curve family and conveyance

The table is plain data (``DEFAULT_TOOLS``) and can be extended / overridden without code: write it once with
``write_tool_table(path)`` and edit the JSON (``<project>_petro/tool_table.json`` is picked up automatically).
Resolutions are TYPICAL published vertical resolutions in metres (editable); the effective resolution of a curve
is max(tool resolution, 2 x sample interval).
"""
import json
import os
import re

import numpy as np

from ..curves import DEFAULT_ALIASES, family_of

# vendor name normalisation (substring of the upper-cased text -> vendor)
VENDORS = [
    ("SCHLUMBERGER", "SLB"), ("SLB", "SLB"), ("D&M", "SLB"), ("ANADRILL", "SLB"), ("GEOQUEST", "SLB"),
    ("BAKER", "Baker Hughes"), ("ATLAS", "Baker Hughes"), ("INTEQ", "Baker Hughes"), ("WESTERN ATLAS", "Baker Hughes"),
    ("HALLIBURTON", "Halliburton"), ("SPERRY", "Halliburton"), ("HLS", "Halliburton"), ("GEARHART", "Halliburton"),
    ("WEATHERFORD", "Weatherford"), ("REEVES", "Weatherford"), ("COMPUTALOG", "Weatherford"),
    ("SCIENTIFIC DRILLING", "Scientific Drilling"),
]

# kind -> typical vertical resolution (m) used when a tool entry gives none
KIND_RES = {
    "gamma": 0.45, "spectral gamma": 0.45, "sp": 2.0, "caliper": 0.15, "density": 0.45, "pe": 0.1,
    "neutron": 0.6, "sonic": 0.6, "laterolog": 0.6, "induction": 1.2, "propagation": 0.6,
    "microresistivity": 0.08, "unknown": 0.6,
}

# family -> kind when nothing better is known
FAMILY_KIND = {"GR": "gamma", "SP": "sp", "CALI": "caliper", "RHOB": "density", "PE": "pe", "NPHI": "neutron",
               "DT": "sonic", "RDEEP": "resistivity", "RSHAL": "resistivity"}

# name: vendor, desc, kind, conveyance, families measured, mnemonics (distinctive), aliases (service tokens,
# DLIS generic / trademark names), res (m), res_by_mnemonic, notes (defaults that matter for interpretation)
DEFAULT_TOOLS = {
    # ---------------- SLB wireline
    "HALS": dict(vendor="SLB", desc="High-resolution Azimuthal Laterolog Sonde", kind="laterolog", conveyance="wireline",
                 families=["RDEEP", "RSHAL"], mnemonics=["HLLD", "HLLS", "HDRS", "HMRS", "HLLG", "HRLD", "HRLS"],
                 aliases=["HALS", "HALS-B", "HILT", "HAL"], res=0.45, res_by_mnemonic={"HRLD": 0.2, "HRLS": 0.2}),
    "HRLA": dict(vendor="SLB", desc="High-Resolution Laterolog Array", kind="laterolog", conveyance="wireline",
                 families=["RDEEP", "RSHAL"], mnemonics=["RLA0", "RLA1", "RLA2", "RLA3", "RLA4", "RLA5", "RT_HRLT", "RXO_HRLT"],
                 aliases=["HRLA", "HRLT", "HRLT-B"], res=0.3),
    "DLL": dict(vendor="SLB", desc="Dual Laterolog", kind="laterolog", conveyance="wireline", families=["RDEEP", "RSHAL"],
                mnemonics=["LLD", "LLS", "LLG"], aliases=["DLL", "DLT", "DLS"], res=0.6),
    "AIT": dict(vendor="SLB", desc="Array Induction Imager", kind="induction", conveyance="wireline",
                families=["RDEEP", "RSHAL"],
                mnemonics=[f"{p}{d}" for p in ("AT", "AF", "AO", "AHT", "AHF", "AHO") for d in ("10", "20", "30", "60", "90")],
                aliases=["AIT", "AIT-H", "AIT-M", "AITH", "AIT-B"], res=0.6,
                res_by_mnemonic={**{f"AF{d}": 0.3 for d in ("10", "20", "30", "60", "90")},
                                 **{f"AHF{d}": 0.3 for d in ("10", "20", "30", "60", "90")},
                                 **{f"AO{d}": 1.2 for d in ("10", "20", "30", "60", "90")},
                                 **{f"AHO{d}": 1.2 for d in ("10", "20", "30", "60", "90")}}),
    "DIL": dict(vendor="SLB", desc="Dual Induction (ILD / ILM / SFL)", kind="induction", conveyance="wireline",
                families=["RDEEP", "RSHAL"], mnemonics=["ILD", "ILM", "IDPH", "IMPH", "SFLU", "SFLA", "SFL"],
                aliases=["DIL", "DIS", "DITE", "DIT", "PI", "ISF", "DISF"], res=2.4,
                res_by_mnemonic={"ILM": 1.5, "IMPH": 1.5, "SFLU": 0.75, "SFLA": 0.75, "SFL": 0.75}),
    "MSFL": dict(vendor="SLB", desc="Micro-Spherically Focused Log", kind="microresistivity", conveyance="wireline",
                 families=["RSHAL"], mnemonics=["MSFL", "MSFC", "SMNO", "SMIN"], aliases=["MSFL", "SRT", "MLL", "ML"],
                 res=0.08),
    "MCFL": dict(vendor="SLB", desc="Micro-Cylindrically Focused Log (Platform Express)", kind="microresistivity",
                 conveyance="wireline", families=["RSHAL"], mnemonics=["RXOZ", "RXO8", "RXOI"],
                 aliases=["MCFL", "HMCF"], res=0.05),
    "TLD": dict(vendor="SLB", desc="Three-detector Lithology Density (Platform Express)", kind="density",
                conveyance="wireline", families=["RHOB", "PE", "CALI"], mnemonics=["RHOZ", "PEFZ", "HDRA", "RHO8", "PEF8", "HCAL", "DRHO_HILT"],
                aliases=["TLD", "PEX", "HILT", "HRDD", "HRMS"], res=0.45,
                res_by_mnemonic={"RHO8": 0.2, "PEF8": 0.05, "PEFZ": 0.05, "HCAL": 0.15}),
    "LDL": dict(vendor="SLB", desc="Litho-Density Log", kind="density", conveyance="wireline", families=["RHOB", "PE", "CALI"],
                mnemonics=["LDL"], aliases=["LDL", "LDT", "LDS", "FDC", "FDT"], res=0.45, res_by_mnemonic={"PEF": 0.1}),
    "CNL": dict(vendor="SLB", desc="Compensated Neutron Log", kind="neutron", conveyance="wireline", families=["NPHI"],
                mnemonics=["NPOR", "NPHI", "TNPH", "NPLE", "TNPL", "HTNP", "NPRL"], aliases=["CNL", "CNT", "CNT-G", "HGNS", "HCNT"],
                res=0.6, notes={"neutron_matrix": "limestone"}),
    "DSI": dict(vendor="SLB", desc="Dipole Shear Sonic Imager", kind="sonic", conveyance="wireline", families=["DT"],
                mnemonics=["DTCO", "DTSM", "DTST", "DT1", "DT2", "DT4P", "DT4S"], aliases=["DSI", "DSST", "DSST-B", "SONICSCANNER", "MSIP"],
                res=1.07),
    "BHC": dict(vendor="SLB", desc="Borehole-Compensated / Long-Spaced Sonic", kind="sonic", conveyance="wireline",
                families=["DT"], mnemonics=["DTL", "DTLN", "DTLF", "ITT", "BCSL"], aliases=["BHC", "LSS", "SLS", "SLT", "SDT", "AS", "SONIC"],
                res=0.6),
    "NGS": dict(vendor="SLB", desc="Natural Gamma-ray Spectrometry", kind="spectral gamma", conveyance="wireline",
                families=["GR"], mnemonics=["SGR", "CGR", "POTA", "THOR", "URAN", "HSGR", "HCGR", "HTHO", "HURA", "HFK"],
                aliases=["NGS", "NGT", "HNGS"], res=0.45),
    "EDTC": dict(vendor="SLB", desc="Enhanced telemetry + gamma ray", kind="gamma", conveyance="wireline", families=["GR"],
                 mnemonics=["GR_EDTC", "ECGR", "EHGR"], aliases=["EDTC", "EDTC-B", "SGT", "HGNS"], res=0.3),
    "AMS": dict(vendor="SLB", desc="Auxiliary Measurement Sonde (mud temperature / resistivity)", kind="unknown",
                conveyance="wireline", families=[], mnemonics=["MTEM", "MRES", "AMTE"], aliases=["AMS"], res=0.6),
    # ---------------- SLB LWD
    "ARC": dict(vendor="SLB", desc="arcVISION array resistivity compensated (LWD)", kind="propagation", conveyance="LWD",
                families=["RDEEP", "RSHAL", "GR"],
                mnemonics=[f"{p}{d}{f}" for p in ("P", "A") for d in ("16", "22", "28", "34", "40") for f in ("H", "L")]
                + ["GR_ARC", "GRAR", "ATMP", "APRS", "ECD_ARC"],
                aliases=["ARC", "ARCVISION", "ARC5", "ARC6", "ARC8", "ARC9"], res=0.3,
                res_by_mnemonic={**{f"A{d}{f}": 0.9 for d in ("16", "22", "28", "34", "40") for f in ("H", "L")},
                                 "GR_ARC": 0.45}),
    "ADN": dict(vendor="SLB", desc="adnVISION azimuthal density neutron (LWD)", kind="density", conveyance="LWD",
                families=["RHOB", "NPHI", "PE", "CALI"], mnemonics=["ROBB", "ROBU", "ROBL", "ROBR", "ROBI", "DRHB", "PEB", "TNPH_ADN", "TNPH"],
                aliases=["ADN", "ADNVISION", "ADN4", "ADN6", "ADN8"], res=0.45, notes={"neutron_matrix": "limestone"}),
    # ---------------- Baker Hughes
    "HDIL": dict(vendor="Baker Hughes", desc="High-Definition Induction Log", kind="induction", conveyance="wireline",
                 families=["RDEEP", "RSHAL"],
                 mnemonics=[f"M{v}R{d}" for v in ("1", "2", "4") for d in ("1", "2", "3", "6", "9", "X")],
                 aliases=["HDIL", "HDI", "DPIL", "DIFL"], res=0.6,
                 res_by_mnemonic={**{f"M1R{d}": 0.3 for d in ("1", "2", "3", "6", "9", "X")},
                                  **{f"M4R{d}": 1.2 for d in ("1", "2", "3", "6", "9", "X")}}),
    "ZDL": dict(vendor="Baker Hughes", desc="Z-Densilog", kind="density", conveyance="wireline", families=["RHOB", "PE", "CALI"],
                mnemonics=["ZDEN", "ZCOR", "ZDNC", "PE_ZDL"], aliases=["ZDL", "ZDEN"], res=0.45),
    "CN": dict(vendor="Baker Hughes", desc="Compensated Neutron (Atlas)", kind="neutron", conveyance="wireline", families=["NPHI"],
               mnemonics=["CNC", "CNCF", "CNCLS", "CNCSS", "CNCDL"], aliases=["CN", "DSN-ATLAS"], res=0.6,
               notes={"neutron_matrix": "limestone (CNCSS = sandstone)"}),
    "XMAC": dict(vendor="Baker Hughes", desc="Cross-dipole Multipole Array Acoustic", kind="sonic", conveyance="wireline",
                 families=["DT"], mnemonics=["DTC_XMAC", "DTS_XMAC", "DTCX", "DTSX"], aliases=["XMAC", "MAC", "DAL", "BCS"], res=1.0),
    "MPR": dict(vendor="Baker Hughes", desc="Multiple Propagation Resistivity (LWD, OnTrak)", kind="propagation", conveyance="LWD",
                families=["RDEEP", "RSHAL"], mnemonics=["RPCEHM", "RPCELM", "RPCSHM", "RPCSLM", "RACEHM", "RACELM", "RACSHM", "RACSLM"],
                aliases=["MPR", "ONTRAK"], res=0.3),
    # ---------------- Halliburton
    "ACRT": dict(vendor="Halliburton", desc="Array Compensated True Resistivity", kind="induction", conveyance="wireline",
                 families=["RDEEP", "RSHAL"], mnemonics=["RT90_ACRT", "RO90", "RO60", "RO30", "RO20", "RO10"], aliases=["ACRT", "HRAI", "HRI"],
                 res=0.6),
    "WAVESONIC": dict(vendor="Halliburton", desc="WaveSonic", kind="sonic", conveyance="wireline", families=["DT"],
                      mnemonics=["DTC_WS", "DTS_WS"], aliases=["WAVESONIC", "WSTT", "FWS"], res=1.0),
    "SDL": dict(vendor="Halliburton", desc="Spectral Density Log", kind="density", conveyance="wireline", families=["RHOB", "PE"],
                mnemonics=["SDL"], aliases=["SDL", "SDLT"], res=0.45),
    "DSN": dict(vendor="Halliburton", desc="Dual-Spaced Neutron", kind="neutron", conveyance="wireline", families=["NPHI"],
                mnemonics=["NPHI_DSN"], aliases=["DSN", "DSNT"], res=0.6),
    "EWR": dict(vendor="Halliburton", desc="Sperry EWR-Phase 4 resistivity (LWD)", kind="propagation", conveyance="LWD",
                families=["RDEEP", "RSHAL"], mnemonics=["SEDP", "SEMP", "SESP", "SEXP", "SEDA", "SEMA", "SESA", "SEXA"],
                aliases=["EWR", "EWR-P4", "EWR-M5"], res=0.3),
}

# generic tools by (family, conveyance) - the fallback
GENERIC = {
    "GR": ("Gamma ray", "gamma"), "SP": ("Spontaneous potential", "sp"), "CALI": ("Caliper", "caliper"),
    "RHOB": ("Density", "density"), "PE": ("Photoelectric factor", "pe"), "NPHI": ("Neutron", "neutron"),
    "DT": ("Sonic", "sonic"), "RDEEP": ("Deep resistivity", "resistivity"), "RSHAL": ("Shallow resistivity", "resistivity"),
}

DESC_KINDS = [  # (regex on the upper-cased description, kind, tool hint)
    (r"LATEROLOG|\bLL[DSG]\b", "laterolog", None), (r"INDUCTION", "induction", None),
    (r"PHASE[- ]SHIFT|ATTENUATION RES|PROPAGATION", "propagation", None),
    (r"SPHERICALLY FOCUSED|MICRO ?LOG|MICRO[- ]NORMAL|MICRO[- ]INVERSE|MICRO RES|FLUSHED ZONE|\bRXO\b", "microresistivity", None),
    (r"\bARC\b", None, "ARC"), (r"ADN", None, "ADN"), (r"\bLWD\b|WHILE DRILLING|\bMWD\b", None, None),
]

LWD_MNEMONICS = re.compile(r"^(ROP|RPM|WOB|TORQ|SPP|ECD|ATMP|APRS)")
WIRELINE_MNEMONICS = {"TENS", "CS", "CVEL", "HTEN", "ETIM", "MARK", "TENSION"}


def _upper_list(v):
    return [str(x).upper() for x in (v or [])]


def tool_table(project=None, path=None):
    """DEFAULT_TOOLS merged with an editable JSON ({name: {...}} - new tools or fields that override)."""
    table = {k: dict(v) for k, v in DEFAULT_TOOLS.items()}
    if path is None and project is not None:
        from .store import petro_dir
        d = petro_dir(project)
        path = os.path.join(d, "tool_table.json") if d else None
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        for name, entry in user.items():
            if name.startswith("_"):
                continue
            base = table.get(name.upper(), table.get(name, {}))
            merged = dict(base)
            merged.update(entry)
            table[name.upper() if name.upper() in table else name] = merged
    for e in table.values():
        e.setdefault("families", [])
        e.setdefault("mnemonics", [])
        e.setdefault("aliases", [])
        e.setdefault("res_by_mnemonic", {})
        e.setdefault("notes", {})
        e.setdefault("res", KIND_RES.get(e.get("kind", "unknown"), 0.6))
        e["mnemonics"] = _upper_list(e["mnemonics"])
        e["aliases"] = _upper_list(e["aliases"])
    return table


def write_tool_table(path):
    """Write the default table as JSON for editing (add tools / mnemonics / resolutions)."""
    d = {"_readme": "GeoHarmonic tool table. Entries here are merged over the built-in defaults by name. "
                    "res = typical vertical resolution in metres; res_by_mnemonic overrides per curve."}
    d.update(DEFAULT_TOOLS)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return path


def vendor_of(text):
    t = (text or "").upper()
    for key, v in VENDORS:
        if key in t:
            return v
    return None


def _tokens(text):
    return [t for t in re.split(r"[^A-Z0-9]+", (text or "").upper()) if t]


class ToolMatcher:
    def __init__(self, table=None):
        self.table = table or tool_table()
        self.by_mnem = {}
        self.by_alias = {}
        for name, e in self.table.items():
            for m in e["mnemonics"]:
                self.by_mnem.setdefault(m, []).append(name)
            for a in e["aliases"] + [name.upper()]:
                self.by_alias.setdefault(a, []).append(name)

    def from_alias_tokens(self, tokens, family, conveyance=None):
        """First tool named by one of the tokens that measures `family` (conveyance must not contradict)."""
        for t in tokens:
            for name in self.by_alias.get(t, []):
                e = self.table[name]
                if family in e["families"] and (conveyance in (None, "unknown") or e["conveyance"] == conveyance):
                    return name
        return None

    def from_mnemonic(self, mnem, family, conveyance=None):
        names = self.by_mnem.get(mnem.upper(), [])
        if not names:
            return None
        good = [n for n in names if conveyance in (None, "unknown") or self.table[n]["conveyance"] == conveyance]
        if not good:
            return None  # e.g. TNPH in an LWD file is not the wireline CNL
        fam_ok = [n for n in (good or names) if family is None or family in self.table[n]["families"] or not self.table[n]["families"]]
        return (fam_ok or good or names)[0]


# ------------------------------------------------------------------------------------------- per file evidence
def file_evidence(src, matcher=None):
    """Evidence from one LasFile / DlisLogs-like object (or a DLIS path): vendor, conveyance, service tokens,
    DLIS channel -> tool map.  Returns a dict."""
    matcher = matcher or ToolMatcher()
    path = getattr(src, "path", str(src))
    ev = {"file": os.path.basename(path), "path": path, "vendor": None, "vendor_source": None, "conveyance": "unknown",
          "conveyance_reason": "", "services": [], "dlis_tools": {}, "dlis_tool_info": {}, "run": None}
    texts = []
    if path.lower().endswith(".dlis"):
        try:
            from ..dlis import DlisFile, ObName
            f = DlisFile(path)
            for lf in f.logical_files:
                info = lf.origin_info()
                pv = vendor_of(str(info.get("PRODUCER-NAME", "")))
                if pv and not ev["vendor"]:
                    ev["vendor"], ev["vendor_source"] = pv, "DLIS ORIGIN PRODUCER-NAME"
                if info.get("RUN-NUMBER") and not ev["run"]:
                    ev["run"] = str(info["RUN-NUMBER"])
                ev["services"] += _tokens(str(info.get("FILE-ID", "")))
                for t in lf.tools:
                    gen = str(t.first("GENERIC-NAME") or "")
                    tm = str(t.first("TRADEMARK-NAME") or "")
                    desc = str(t.first("DESCRIPTION") or "")
                    chans = []
                    for c in t.get("CHANNELS") or []:
                        chans.append((c.id if isinstance(c, ObName) else str(c)).upper())
                    ev["dlis_tool_info"][t.name.id] = {"generic": gen, "trademark": tm, "description": desc}
                    for c in chans:
                        ev["dlis_tools"].setdefault(c, t.name.id)
                    texts.append(desc)
        except Exception as e:  # unreadable DLIS: fall back to the curves
            ev["error"] = str(e)
    well = getattr(src, "well", {}) or {}
    params = getattr(src, "params", {}) or {}

    def hv(key):
        v = well.get(key) or params.get(key)
        if not v:
            return ""
        return " ".join(str(x) for x in v[:1] + v[2:3] if x)  # value + description (LAS 1.2 puts values there)

    srvc = hv("SRVC")
    if not ev["vendor"] and vendor_of(srvc):
        ev["vendor"], ev["vendor_source"] = vendor_of(srvc), "LAS SRVC"
    for k in ("OS", "OS1", "OS2", "OS3", "SERV", "SVCS", "TOOL"):
        ev["services"] += _tokens(well.get(k, ("",))[0] if well.get(k) else "")
        ev["services"] += _tokens(params.get(k, ("",))[0] if params.get(k) else "")
    if not ev["run"]:
        r = params.get("RUN") or well.get("RUN")
        ev["run"] = r[0] if r and r[0] else None
    stem = os.path.splitext(os.path.basename(path))[0]
    ev["services"] = list(dict.fromkeys(_tokens(stem) + ev["services"]))
    curves = getattr(src, "curves", []) or []
    mn = {c.mnemonic.upper() for c in curves}
    descs = " | ".join(getattr(c, "description", "") or "" for c in curves).upper()
    texts.append(descs)
    lwd, wl = [], []
    if any(LWD_MNEMONICS.match(m) for m in mn):
        lwd.append("drilling channels (ROP/ECD/...)")
    if re.search(r"BIT DEPTH|WHILE DRILLING|\bLWD\b|\bMWD\b", descs):
        lwd.append("LWD wording in curve descriptions")
    if re.search(r"D&M|ANADRILL|SPERRY|INTEQ|DRILLING", srvc.upper()):
        lwd.append(f"service company '{srvc.strip()}'")
    wl_mn = {m for e in matcher.table.values() if e["conveyance"] != "LWD" for m in e["mnemonics"]}
    lwd_tools = [n for n in matcher.table if matcher.table[n]["conveyance"] == "LWD"
                 and ((set(matcher.table[n]["mnemonics"]) - wl_mn) & mn or n in ev["services"])]
    if lwd_tools:
        lwd.append("LWD tool mnemonics (" + ", ".join(sorted(lwd_tools)) + ")")
    if mn & WIRELINE_MNEMONICS:
        wl.append("cable tension / speed channels")
    if ev["dlis_tools"] and not lwd:
        wl.append("wireline DLIS tool string")
    if lwd and (len(lwd) > len(wl) or (len(lwd) == len(wl) and not ev["dlis_tools"])):
        ev["conveyance"], ev["conveyance_reason"] = "LWD", "; ".join(lwd)
    elif wl:
        ev["conveyance"], ev["conveyance_reason"] = "wireline", "; ".join(wl)
    if not ev["vendor"]:
        v = vendor_of(" ".join(texts))
        if v:
            ev["vendor"], ev["vendor_source"] = v, "text in descriptions"
    return ev


def sample_interval(depth):
    d = np.asarray(depth, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    s = np.abs(np.diff(d))
    s = s[s > 0]
    return float(np.median(s)) if s.size else float("nan")


def identify_curve(curve, ev, matcher, aliases=DEFAULT_ALIASES):
    """Tool, vendor, kind, conveyance, resolution for one LasCurve-like curve with its file evidence."""
    mnem = curve.mnemonic.upper()
    fam, _ = family_of(mnem, aliases)
    conv = ev["conveyance"]
    tool = None
    how = ""
    # 1 DLIS tool object listing this channel
    tname = ev["dlis_tools"].get(mnem)
    if tname:
        ti = ev["dlis_tool_info"].get(tname, {})
        toks = _tokens(ti.get("generic")) + _tokens(ti.get("trademark")) + _tokens(tname)
        tool = matcher.from_alias_tokens([ti.get("generic", "").upper(), ti.get("trademark", "").upper()] + toks, fam)
        if tool is None:
            tool = matcher.from_mnemonic(mnem, fam)
        how = f"DLIS tool {tname} ({ti.get('description', '')})".strip()
        if tool is None:
            tool = f"{ti.get('trademark') or tname} (DLIS)"
    # 2 distinctive mnemonic
    if tool is None:
        t = matcher.from_mnemonic(mnem, fam, conv)
        if t is not None and (fam is None or fam in matcher.table[t]["families"] or not matcher.table[t]["families"]):
            tool, how = t, "mnemonic table"
    # 3 service tokens in the file name / OS lines
    if tool is None and fam is not None:
        t = matcher.from_alias_tokens(ev["services"], fam, conv)
        if t is not None:
            tool, how = t, "service in file name / header"
    # 4 description keywords
    kind = None
    desc = (getattr(curve, "description", "") or "").upper()
    if tool is None:
        for rx, k, hint in DESC_KINDS:
            if re.search(rx, desc):
                if hint and hint in matcher.table and (fam is None or fam in matcher.table[hint]["families"]):
                    tool, how = hint, "curve description"
                    break
                if k and kind is None:
                    kind = k
    entry = matcher.table.get(tool) if tool else None
    if entry is not None:
        if fam in ("RDEEP", "RSHAL") or fam is None:
            kind = entry["kind"] if entry["kind"] != "unknown" or fam is None else FAMILY_KIND.get(fam, "unknown")
        elif fam == "GR" and entry["kind"] == "spectral gamma":
            kind = "spectral gamma"
        else:
            kind = FAMILY_KIND.get(fam, entry["kind"])
    if kind is None:
        kind = FAMILY_KIND.get(fam, "unknown") if fam else "unknown"
    if tool is None and fam is not None:
        gname, gkind = GENERIC[fam]
        tool = f"{gname} ({conv})" if conv != "unknown" else gname
        how = "generic for the curve family"
        if kind == "resistivity":
            kind = gkind
    if kind == "resistivity":  # still unknown laterolog vs induction
        kind = "resistivity (type unknown)"
    vendor = (entry["vendor"] if entry else None) if not ev["vendor"] else ev["vendor"]
    if entry and ev["vendor"] and entry["vendor"] != ev["vendor"] and how == "mnemonic table":
        how += f" (vendor in file says {ev['vendor']})"
    if entry and conv == "unknown":
        conv = entry["conveyance"]
    res = None
    if entry:
        res = entry["res_by_mnemonic"].get(mnem, entry["res"])
    if res is None:
        res = KIND_RES.get(kind.split(" (")[0], KIND_RES["unknown"])
    smp = sample_interval(curve.depth)
    eff = max(res, 2 * smp) if np.isfinite(smp) else res
    top, base = curve.valid_range()
    return {
        "mnemonic": mnem, "family": fam, "unit": curve.unit, "description": getattr(curve, "description", ""),
        "file": ev["file"], "column": getattr(curve, "index", None), "tool": tool, "tool_desc": entry["desc"] if entry else "",
        "vendor": vendor or "unknown", "kind": kind, "conveyance": conv, "how": how, "sample_m": smp,
        "tool_res_m": float(res), "eff_res_m": float(eff), "undersampled": bool(np.isfinite(smp) and 2 * smp > 1.1 * res),
        "top": top, "base": base, "notes": dict(entry["notes"]) if entry else {},
    }


def open_sources(well):
    """LasFile / DlisLogs objects for every log file of a project well (errors reported, not raised)."""
    from ..las import LasFile
    out, errors = [], []
    for pth in well.las_paths:
        try:
            if pth.lower().endswith(".dlis"):
                from ..dlis import DlisLogs
                out.append(DlisLogs(pth))
            else:
                out.append(LasFile(pth))
        except Exception as e:
            errors.append(f"{os.path.basename(pth)}: {e}")
    return out, errors


def identify_well(project, name, table=None, sources=None):
    """Per-curve tool identification for a well.  Returns {"files": [evidence], "curves": [info], "errors": []}."""
    w = project.wells[name]
    matcher = ToolMatcher(table or tool_table(project))
    if sources is None:
        sources, errors = open_sources(w)
    else:
        errors = []
    files, curves = [], []
    for src in sources:
        ev = file_evidence(src, matcher)
        fe = {k: v for k, v in ev.items() if k not in ("dlis_tools", "dlis_tool_info")}
        fe["dlis_tools"] = sorted(set(ev["dlis_tool_info"]))
        files.append(fe)
        for c in src.curves:
            if c.n_valid < 2:
                continue
            curves.append(identify_curve(c, ev, matcher, project.aliases))
    return {"well": name, "files": files, "curves": curves, "errors": errors}


_SRC_RX = re.compile(r"^(\S+) \[(.+) col (\d+)\]")


def used_curves(well, ident):
    """For each composited family: the identified curves the composite actually used (from well.sources)."""
    out = {}
    by_key = {(c["mnemonic"], c["file"].lower(), c["column"]): c for c in ident["curves"]}
    for fam, lst in (well.sources or {}).items():
        if fam.startswith("_"):
            continue
        for s in lst:
            m = _SRC_RX.match(s)
            if not m:
                continue
            c = by_key.get((m.group(1).upper(), m.group(2).lower(), int(m.group(3))))
            if c is not None:
                out.setdefault(fam, []).append(c)
    return out
