"""QGIS plugin 'GeoHarmonic Link' (PLAN 21h).  Install: Plugins > Manage and Install Plugins > Install from ZIP
(zip made by dev/build_qgis_plugin.py)."""


def classFactory(iface):  # noqa: N802 (QGIS API name)
    from .plugin import GeoHarmonicLinkPlugin
    return GeoHarmonicLinkPlugin(iface)
