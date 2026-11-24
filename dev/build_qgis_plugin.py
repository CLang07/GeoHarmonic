"""Build the QGIS plugin ZIP (Plugins > Manage and Install Plugins > Install from ZIP).

    python dev/build_qgis_plugin.py            -> dev/geoharmonic_link.zip  (not committed)
"""
import os
import sys
import zipfile

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "..", "tools", "QGIS Plugin", "geoharmonic_link")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "geoharmonic_link.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith((".py", ".txt", ".png", ".qml")):
                p = os.path.join(root, f)
                z.write(p, os.path.join("geoharmonic_link", os.path.relpath(p, src)))
print("wrote", out)
