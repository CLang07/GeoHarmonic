"""Seismic colour tables as 256x4 uint8 lookup tables."""
import numpy as np

_STOPS = {
    "Seismic (red-white-blue)": [(0.0, (0, 0, 140)), (0.25, (40, 90, 255)), (0.5, (255, 255, 255)),
                                 (0.75, (255, 60, 40)), (1.0, (140, 0, 0))],
    "Grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "Red-white-black": [(0.0, (190, 0, 0)), (0.5, (255, 255, 255)), (1.0, (0, 0, 0))],
    "Petrel-like": [(0.0, (160, 255, 255)), (0.15, (0, 70, 220)), (0.42, (60, 60, 60)), (0.5, (200, 200, 200)),
                    (0.58, (90, 60, 50)), (0.85, (230, 40, 0)), (1.0, (255, 255, 120))],
    "Rainbow (attributes)": [(0.0, (48, 18, 59)), (0.25, (40, 160, 230)), (0.5, (120, 250, 80)),
                             (0.75, (250, 180, 40)), (1.0, (122, 4, 3))],
}
NAMES = list(_STOPS)


def lut(name, reverse=False):
    stops = _STOPS.get(name, _STOPS[NAMES[0]])
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], float)
    x = np.linspace(0, 1, 256)
    out = np.empty((256, 4), np.uint8)
    for k in range(3):
        out[:, k] = np.interp(x, pos, col[:, k]).astype(np.uint8)
    out[:, 3] = 255
    return out[::-1].copy() if reverse else out
