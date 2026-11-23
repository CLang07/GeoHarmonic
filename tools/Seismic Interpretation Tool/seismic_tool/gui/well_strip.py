"""Well strip / geosteering view: log(s) vs MD, TVDSS vs MD with the zone top / base under the wellbore, and the
seismic section along the wellbore (time vs MD) with the well's time path.  Data from laterals.strip_data."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from . import colormaps  # noqa: E402

LOG_COLORS = [(0, 128, 0), (200, 0, 0), (0, 0, 200), (120, 60, 0)]


class WellStrip(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel("")
        lay.addWidget(self.title)
        self.p_log = pg.PlotWidget()
        self.p_depth = pg.PlotWidget()
        self.p_seis = pg.PlotWidget()
        for p, lab in ((self.p_log, "log"), (self.p_depth, "TVDSS (m)"), (self.p_seis, "TWT (ms)")):
            p.setBackground("w")
            p.setLabel("left", lab)
            lay.addWidget(p, 1)
        self.p_seis.setLabel("bottom", "MD (m)")
        self.p_depth.invertY(True)
        self.p_seis.invertY(True)
        self.p_depth.setXLink(self.p_log)
        self.p_seis.setXLink(self.p_log)
        self.data = None
        self.resize(900, 800)

    def show_data(self, name, d, lateral_only=True):
        """d: laterals.strip_data output."""
        self.data = d
        md = d["md"]
        rng = slice(None)
        if lateral_only:
            # start the display a little before the heel (where the well gets near horizontal)
            dz = np.abs(np.gradient(d["z"]))
            dmd = np.gradient(md)
            horiz = np.nonzero(dz < 0.5 * np.maximum(dmd, 1e-9))[0]
            if len(horiz):
                rng = slice(max(0, horiz[0] - 20), None)
        self.title.setText(f"{name}: MD {md[rng][0]:.0f} - {md[-1]:.0f} m"
                           + (f", {100 * np.nanmean(d['in_zone'][rng]):.0f}% in zone" if "in_zone" in d else ""))
        for p in (self.p_log, self.p_depth, self.p_seis):
            p.clear()
        for k, (n, v) in enumerate(d["logs"].items()):
            self.p_log.plot(md[rng], v[rng], pen=pg.mkPen(LOG_COLORS[k % len(LOG_COLORS)], width=1.5), name=n)
        self.p_depth.plot(md[rng], d["z"][rng], pen=pg.mkPen("k", width=2))
        for key, col in (("top", (200, 0, 0)), ("base", (0, 0, 200))):
            if key in d:
                self.p_depth.plot(md[rng], d[key][rng], pen=pg.mkPen(col, width=1.5))
        if "section" in d:
            sec = d["section"][rng]
            t = d["times"]
            # the app uses imageAxisOrder="row-major": rows are y (time), so transpose (n_md, ns) -> (ns, n_md)
            img = pg.ImageItem(np.nan_to_num(sec, nan=0.0).T)
            clip = float(np.nanpercentile(np.abs(sec), 99)) if np.isfinite(sec).any() else 1.0
            img.setLookupTable(colormaps.lut(colormaps.NAMES[0]))
            img.setLevels([-clip, clip])
            m0, m1 = md[rng][0], md[rng][-1]
            img.setRect(QtCore.QRectF(m0, t[0], max(m1 - m0, 1.0), t[-1] - t[0]))  # array axis 0 = MD (x)
            self.p_seis.addItem(img)
            if "twt" in d:
                self.p_seis.plot(md[rng], d["twt"][rng], pen=pg.mkPen((255, 200, 0), width=2))
        return rng
