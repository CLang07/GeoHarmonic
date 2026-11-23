"""Write the synthetic engineering + production data set (completions.csv, completions_metric.csv,
production.csv) to a folder:  python dev/make_synthetic_engineering.py OUT_FOLDER"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "Seismic Interpretation Tool"))
from seismic_tool.analytics.synthetic import make  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_engineering"
    t = make(out)
    print(f"wrote {out}: {len(t['api'])} wells")
