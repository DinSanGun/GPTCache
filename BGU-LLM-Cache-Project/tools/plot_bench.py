import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    ("mean_ms", "Mean latency (ms)"),
    ("p95_ms", "p95 latency (ms)"),
    ("p99_ms", "p99 latency (ms)"),
    ("throughput_rps", "Throughput (req/s)"),
    ("sys_cpu_pct_mean", "CPU util mean (%)"),
    ("sys_rss_mb_peak", "Memory RSS peak (MB)"),
    ("hit_rate_warm", "Warm hit rate"),
    ("cost_saved_total", "Cost saved (total)"),
    # ("sys_gpu_util_pct_mean","GPU util mean (%)")  # include if you add GPU metrics
]

def load_runs(run_dirs):
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        s = rd / "summary.csv"
        if s.is_file():
            df = pd.read_csv(s)
            if not df.empty:
                row = df.iloc[0].to_dict()
                # policy name from path tail (or embedded in row["cache"])
                policy = Path(rd).name.split("_", 1)[0].upper()
                row["policy"] = policy
                row["run_dir"] = rd.as_posix()
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def plot_metric(df, key, title, outpng):
    plt.figure()
    sub = df[["policy", key]].copy()
    sub = sub.set_index("policy")
    sub.plot(kind="bar", legend=False, ylabel=title, rot=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Paths to run dirs (each must contain summary.csv)")
    ap.add_argument("--out", required=True, help="Output folder for PNGs and combined CSV")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.runs)
    if df.empty:
        raise SystemExit("No valid summary.csv found in provided run dirs.")

    # save combined CSV
    combined_csv = out / "comparison_summary.csv"
    df.to_csv(combined_csv, index=False, encoding="utf-8")

    # plots
    for key, title in METRICS:
        if key in df.columns:
            plot_metric(df, key, title, out / f"{key}.png")

    # simple console print
    cols = ["policy"] + [k for k, _ in METRICS if k in df.columns]
    print(df[cols].sort_values("policy").to_string(index=False))

if __name__ == "__main__":
    main()
