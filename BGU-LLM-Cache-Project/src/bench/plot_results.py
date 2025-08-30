#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv_labeled(path: str | Path, label: str | None) -> Tuple[str, pd.DataFrame]:
    p = Path(path)
    df = pd.read_csv(p)
    name = label or p.stem
    return name, df

def ensure_out(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.dropna()

def percentile(vals: np.ndarray, p: float) -> float:
    xs = np.sort(np.array(vals, dtype=float))
    if xs.size == 0:
        return 0.0
    k = int((xs.size - 1) * p)
    return float(xs[k])

def plot_percentiles_mean(datasets: List[Tuple[str, pd.DataFrame]], outdir: Path) -> None:
    # For each dataset, compute mean, p95, p99 from latency columns
    for name, df in datasets:
        col = "lat_ms" if "lat_ms" in df.columns else ("e2e_ms" if "e2e_ms" in df.columns else None)
        if not col: 
            continue
        xs = safe_float_series(df, col).to_numpy()
        if xs.size == 0: 
            continue
        mean = float(xs.mean())
        p95 = percentile(xs, 0.95)
        p99 = percentile(xs, 0.99)
        plt.figure()
        plt.bar(["mean","p95","p99"], [mean, p95, p99])
        plt.ylabel("Latency (ms)")
        plt.title(f"Latency (mean/p95/p99): {name}")
        plt.tight_layout()
        plt.savefig(outdir / f"percentiles_{name}.png", dpi=160)
        plt.close()

def hit_rate(df: pd.DataFrame) -> float:
    if "hit" not in df.columns or len(df) == 0:
        return 0.0
    s = pd.to_numeric(df["hit"], errors="coerce").fillna(0)
    return float(s.sum()) / max(1, len(s))

def plot_hit_rate(datasets: List[Tuple[str, pd.DataFrame]], outdir: Path) -> None:
    labels, vals = [], []
    for name, df in datasets:
        if "hit" in df.columns:
            labels.append(name)
            vals.append(hit_rate(df) * 100.0)
    if not labels:
        return
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Hit rate (%)")
    plt.title("Cache Hit Rate")
    plt.tight_layout()
    plt.savefig(outdir / "hit_rate.png", dpi=160)
    plt.close()

def virtual_throughput_from_latencies(df: pd.DataFrame, col: str) -> float:
    lats = safe_float_series(df, col)
    total_s = lats.sum() / 1000.0
    n = len(lats)
    return (n / total_s) if total_s > 0 else 0.0

def plot_throughput(datasets: List[Tuple[str, pd.DataFrame]], outdir: Path) -> None:
    labels, vals = [], []
    for name, df in datasets:
        # Prefer 'throughput_rps' if we're passed a summary CSV
        if "throughput_rps" in df.columns and len(df) == 1:
            labels.append(name)
            vals.append(float(df["throughput_rps"].iloc[0]))
            continue
        # else infer from latencies (approximate, assumes serial calls)
        col = "lat_ms" if "lat_ms" in df.columns else ("e2e_ms" if "e2e_ms" in df.columns else None)
        if col:
            labels.append(name)
            vals.append(virtual_throughput_from_latencies(df, col))
    if not labels:
        return
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Requests per second")
    plt.title("Throughput")
    plt.tight_layout()
    plt.savefig(outdir / "throughput.png", dpi=160)
    plt.close()

def plot_sys_timeseries(sys_csv: Path, outdir: Path) -> None:
    if not sys_csv.exists():
        return
    df = pd.read_csv(sys_csv)
    if not {"ts","cpu_pct","rss_mb"}.issubset(df.columns):
        return
    # CPU
    plt.figure()
    df.plot(x="ts", y="cpu_pct", legend=False)
    plt.xlabel("timestamp (s)")
    plt.ylabel("CPU (%)")
    plt.title("CPU utilization over time")
    plt.tight_layout()
    plt.savefig(outdir / "cpu_timeseries.png", dpi=160)
    plt.close()
    # Memory
    plt.figure()
    df.plot(x="ts", y="rss_mb", legend=False)
    plt.xlabel("timestamp (s)")
    plt.ylabel("RSS (MB)")
    plt.title("Memory (RSS) over time")
    plt.tight_layout()
    plt.savefig(outdir / "memory_timeseries.png", dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot CSVs: mean/p95/p99, throughput, hit rate, CPU/mem (if present).")
    ap.add_argument("--csv", nargs="+", required=True, help="List of CSV files to visualize (detail.csv or summary.csv).")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels matching the CSV order.")
    ap.add_argument("--out", default="plots", help="Output directory for images.")
    ap.add_argument("--sys-csv", default=None, help="Optional sys_detail.csv to plot CPU/memory.")
    args = ap.parse_args()

    outdir = Path(args.out)
    ensure_out(outdir)

    datasets: List[Tuple[str, pd.DataFrame]] = []
    for i, path in enumerate(args.csv):
        label = (args.labels[i] if args.labels and i < len(args.labels) else None)
        datasets.append(load_csv_labeled(path, label))

    # Plots requested by user
    plot_percentiles_mean(datasets, outdir)  # mean/p95/p99
    plot_throughput(datasets, outdir)
    plot_hit_rate(datasets, outdir)

    # Optional system timeseries
    if args.sys_csv:
        plot_sys_timeseries(Path(args.sys_csv), outdir)

    print(f"Saved plots in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
