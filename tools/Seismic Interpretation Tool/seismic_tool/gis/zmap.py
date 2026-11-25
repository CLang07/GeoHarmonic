"""ZMAP+ grid read / write (PLAN 31i) - the plain-text grid format Petrel, Kingdom, GeoGraphix, Zmap and OpendTect
all import.  numpy only.

    ! comments
    @<name> HEADER, GRID, <nodes per line>
    <field width>, <null value>, <null string>, <decimals>, <start column>
    <rows>, <columns>, <xmin>, <xmax>, <ymin>, <ymax>
    0.0, 0.0, 0.0
    @
    values column by column (west to east), each column from the top row (ymax) down to ymin
"""
import numpy as np

NULL = 1e30


def write_zmap(path, grid, name=None, per_line=5, width=15, decimals=7, comment=""):
    """MapGrid (an ilxl grid is resampled to an axis-aligned XY grid first) -> ZMAP+ file."""
    g = grid.to_xy() if getattr(grid, "kind", "xy") != "xy" else grid
    z = np.asarray(g.z, float)
    ny, nx = z.shape
    xmin, ymin = float(g.x0), float(g.y0)
    xmax, ymax = xmin + g.dx * (nx - 1), ymin + g.dy * (ny - 1)
    nm = (name or getattr(g, "label", "") or "GRID").replace(",", " ")[:40]
    vals = z[::-1, :].T.ravel()  # columns west -> east, each top (ymax) -> bottom
    vals = np.where(np.isfinite(vals), vals, NULL)
    with open(path, "w") as f:
        f.write(f"! GeoHarmonic ZMAP+ export{(': ' + comment) if comment else ''}\n")
        f.write(f"! {nm} ({getattr(g, 'unit', '')})\n")
        f.write(f"@{nm} HEADER, GRID, {per_line}\n")
        f.write(f"{width}, {NULL:.0E}, , {decimals}, 1\n")
        f.write(f"{ny}, {nx}, {xmin:.4f}, {xmax:.4f}, {ymin:.4f}, {ymax:.4f}\n")
        f.write("0.0, 0.0, 0.0\n@\n")
        fmt = f"{{:{width}.{decimals}g}}"
        # each column starts on a new line (ZMAP+ convention)
        for j in range(nx):
            col = vals[j * ny:(j + 1) * ny]
            for k in range(0, ny, per_line):
                f.write("".join(fmt.format(v) for v in col[k:k + per_line]) + "\n")
    return path


def read_zmap(path):
    from ..maps import MapGrid
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    k = 0
    while lines[k].lstrip().startswith("!") or not lines[k].strip():
        k += 1
    head = lines[k]
    if not head.startswith("@"):
        raise ValueError("ZMAP+: no '@' header line")
    name = head[1:].split(",")[0].replace("HEADER", "").strip()
    p1 = [s.strip() for s in lines[k + 1].split(",")]
    null = float(p1[1]) if p1[1] else NULL
    p2 = [float(s) for s in lines[k + 2].split(",")[:6]]
    ny, nx = int(p2[0]), int(p2[1])
    xmin, xmax, ymin, ymax = p2[2:6]
    j = k + 3
    while not lines[j].strip().startswith("@"):
        j += 1
    vals = np.array(" ".join(lines[j + 1:]).split(), float)
    if vals.size < nx * ny:
        raise ValueError(f"ZMAP+: {vals.size} values for {ny} x {nx} nodes")
    vals = vals[:nx * ny]
    vals[np.abs(vals - null) <= 1e-6 * abs(null)] = np.nan
    z = vals.reshape(nx, ny).T[::-1, :]
    dx = (xmax - xmin) / max(1, nx - 1)
    dy = (ymax - ymin) / max(1, ny - 1)
    return MapGrid(z, "xy", "", name, None, xmin, ymin, dx, dy)
