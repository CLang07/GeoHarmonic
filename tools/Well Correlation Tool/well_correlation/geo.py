"""Coordinates and well-path geometry (numpy only, no pyproj needed).

NZTM2000 (EPSG:2193): Transverse Mercator on GRS80, lon0 173E, k0 0.9996,
false easting 1 600 000 m, false northing 10 000 000 m.
"""
import math
import re
import numpy as np

_A = 6378137.0
_F = 1 / 298.257222101
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def _meridian_arc(phi):
    e2 = _E2
    e4, e6 = e2 * e2, e2 * e2 * e2
    return _A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
                 - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
                 + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
                 - (35 * e6 / 3072) * math.sin(6 * phi))


def tm_forward(lat, lon, lon0=173.0, k0=0.9996, fe=1600000.0, fn=10000000.0):
    """Lat/lon (degrees) -> (easting, northing) in metres. Default = NZTM2000."""
    phi, lam = math.radians(lat), math.radians(lon - lon0)
    s, c = math.sin(phi), math.cos(phi)
    t = math.tan(phi)
    n = _A / math.sqrt(1 - _E2 * s * s)
    T, C = t * t, _EP2 * c * c
    A = lam * c
    M = _meridian_arc(phi)
    x = k0 * n * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * _EP2) * A ** 5 / 120)
    y = k0 * (M + n * t * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                          + (61 - 58 * T + T * T + 600 * C - 330 * _EP2) * A ** 6 / 720))
    return fe + x, fn + y


def min_curvature(md, inc, azi):
    """Minimum-curvature: returns tvd, dN, dE (same length as md), relative to first station."""
    md, inc, azi = (np.asarray(a, float) for a in (md, inc, azi))
    i1, i2 = np.radians(inc[:-1]), np.radians(inc[1:])
    a1, a2 = np.radians(azi[:-1]), np.radians(azi[1:])
    dmd = np.diff(md)
    cosdl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
    dl = np.arccos(np.clip(cosdl, -1, 1))
    rf = np.where(dl > 1e-9, 2 / np.where(dl > 1e-9, dl, 1) * np.tan(dl / 2), 1.0)
    dn = dmd / 2 * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    de = dmd / 2 * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    dv = dmd / 2 * (np.cos(i1) + np.cos(i2)) * rf
    z = np.concatenate([[0], np.cumsum(dv)])
    n = np.concatenate([[0], np.cumsum(dn)])
    e = np.concatenate([[0], np.cumsum(de)])
    return md[0] + z, n, e  # first station assumed vertical above it


def read_survey(path):
    """Read MD / INC / AZI from a CSV/TXT survey.

    Accepts a simple 3-column CSV (MD,INC,AZI with or without header) or a Schlumberger
    style listing where each row is: seq MD INC AZI ...  Returns arrays or None.
    """
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            toks = re.split(r"[,\s;\t]+", line.strip())
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except ValueError:
                    break
            if len(nums) >= 4 and float(nums[0]).is_integer() and nums[1] >= 0 and 0 <= nums[2] <= 180:
                rows.append(nums[1:4])  # seq md inc azi
            elif len(nums) == 3 and 0 <= nums[1] <= 180:
                rows.append(nums)
    if len(rows) < 2:
        return None
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    if a[0, 0] > 0:  # add a vertical tie-in at surface
        a = np.vstack([[0, 0, 0], a])
    return a[:, 0], a[:, 1], a[:, 2]
