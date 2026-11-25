"""Seismic colour tables as 256x4 uint8 lookup tables.

Since PLAN 40b this is a thin front for `seismic_tool.colortables` (the shared library: seismic bars incl.
Kingdom-style / high contrast, perceptual viridis / cividis / magma-style tables generated in OKLab, colourblind-safe
flags, custom tables saved in the project).  The first five names and their colours are unchanged."""
import numpy as np

from .. import colortables as CT

_STOPS = {k: v for k, v in CT.LIBRARY.items()}
NAMES = CT.names()


def refresh_names(project=None):
    """Re-list the tables (after a project with custom tables opens or one is saved)."""
    NAMES[:] = CT.names(project)
    return NAMES


def lut(name, reverse=False):
    t = CT.get(name) or CT.get(NAMES[0])
    out = t.lut(apply_display=False)
    return out[::-1].copy() if reverse else out
