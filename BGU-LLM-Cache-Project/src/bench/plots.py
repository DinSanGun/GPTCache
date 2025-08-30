from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.bench.plots <csv paths ...>")
        sys.exit(1)

    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists(): 
            print(f"Skip missing {p}")
            continue
        df = pd.read_csv(p)
        if "lat_ms" in df.columns:
            plt.figure()
            df["lat_ms"].plot(kind="hist", bins=30, title=f"Latency histogram: {p.name}")
            out = p.with_suffix(".png")
            plt.savefig(out, bbox_inches="tight")
            print("Saved plot:", out)

if __name__ == "__main__":
    main()
