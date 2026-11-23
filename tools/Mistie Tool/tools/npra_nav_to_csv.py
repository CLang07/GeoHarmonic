"""Convert the USGS NPRA SEG-P1 navigation (NPRANAV.SEG) to the tool's nav CSV (line,cdp,x,y in feet).
The projected x/y columns in that file run into each other, so lat/lon are re-projected here
(local equirectangular about 70N 156W - fine for finding ties, not for survey-grade work).
usage: python tools/npra_nav_to_csv.py NPRANAV.SEG npra_nav.csv"""
import re, sys, math, csv

pat = re.compile(r"^\s*(\S+)\s+(\d+)\s+(-?\d+\.\d+)\s*(-?\d+\.\d+)")
lat0, lon0, R = 70.0, -156.0, 20925646.0
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["line", "cdp", "x", "y"])
    for ln in open(src, encoding="latin-1"):
        m = pat.match(ln)
        if m:
            name, cdp, lat, lon = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
            x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
            y = math.radians(lat - lat0) * R
            w.writerow([name, cdp, f"{x:.1f}", f"{y:.1f}"])
print("wrote", dst)
