#!/usr/bin/env python
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class RunData:
    name: str
    run_dir: Path
    # aggregate metrics (prefer summary.csv)
    mean_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    hit_rate_pct: float  # 0..100
    # raw dfs (optional)
    detail_df: Optional[pd.DataFrame]
    summary_df: Optional[pd.DataFrame]
    sys_df: Optional[pd.DataFrame]

def ensure_out(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def _safe_num(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _percentile(vals: np.ndarray, p: float) -> float:
    xs = np.sort(np.array(vals, dtype=float))
    if xs.size == 0:
        return 0.0
    k = int((xs.size - 1) * p)
    return float(xs[k])

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()

def _compute_from_detail(detail: Optional[pd.DataFrame]) -> Tuple[float,float,float,float,float]:
    if detail is None or detail.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    # prefer lat_ms; fallback to e2e_ms for legacy
    col = "lat_ms" if "lat_ms" in detail.columns else ("e2e_ms" if "e2e_ms" in detail.columns else None)
    if col is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xs = _safe_series(detail, col).to_numpy()
    if xs.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean_ms = float(xs.mean())
    p95_ms = _percentile(xs, 0.95)
    p99_ms = _percentile(xs, 0.99)
    total_s = xs.sum()/1000.0
    throughput = (len(xs)/total_s) if total_s>0 else 0.0
    hit_pct = 0.0
    if "hit" in detail.columns and len(detail) > 0:
        hits = pd.to_numeric(detail["hit"], errors="coerce").fillna(0)
        hit_pct = float(hits.sum())/max(1,len(hits)) * 100.0
    return mean_ms, p95_ms, p99_ms, throughput, hit_pct

def _load_run(csv_path: Path, name: Optional[str]) -> RunData:
    csv_path = Path(csv_path)
    run_dir = csv_path.parent
    # try to open both summary and detail regardless of the passed file
    detail_df = None
    summary_df = None
    if (run_dir / "detail.csv").exists():
        try: detail_df = pd.read_csv(run_dir / "detail.csv")
        except Exception: detail_df = None
    if (run_dir / "summary.csv").exists():
        try: summary_df = pd.read_csv(run_dir / "summary.csv")
        except Exception: summary_df = None

    # prefer summary for aggregates
    if summary_df is not None and not summary_df.empty:
        row = summary_df.iloc[0].to_dict()
        mean_ms = _safe_num(row.get("mean_ms", 0))
        p95_ms  = _safe_num(row.get("p95_ms", 0))
        p99_ms  = _safe_num(row.get("p99_ms", 0))
        throughput = _safe_num(row.get("throughput_rps", 0))
        hit_rate_pct = _safe_num(row.get("hit_rate", 0)) * (100.0 if row.get("hit_rate", None) is not None and row.get("hit_rate") <= 1.0 else 1.0)
    else:
        mean_ms, p95_ms, p99_ms, throughput, hit_rate_pct = _compute_from_detail(detail_df)

    # sys df (optional)
    sys_df = None
    if (run_dir / "sys_detail.csv").exists():
        try: sys_df = pd.read_csv(run_dir / "sys_detail.csv")
        except Exception: sys_df = None

    return RunData(
        name=name or csv_path.stem,
        run_dir=run_dir,
        mean_ms=mean_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        throughput_rps=throughput,
        hit_rate_pct=hit_rate_pct,
        detail_df=detail_df,
        summary_df=summary_df,
        sys_df=sys_df,
    )

def plot_bars(xs: List[str], ys: List[float], ylabel: str, title: str, outpath: Path):
    plt.figure()
    plt.bar(xs, ys)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_cpu_mem(run_datas: List[RunData], outdir: Path):
    # Make CPU/mem plots for each run that has sys_df
    for r in run_datas:
        if r.sys_df is None or r.sys_df.empty:
            continue
        df = r.sys_df
        if not {"ts","cpu_pct","rss_mb"}.issubset(df.columns):
            continue
        # CPU
        plt.figure()
        df.plot(x="ts", y="cpu_pct", legend=False)
        plt.xlabel("timestamp (s)")
        plt.ylabel("CPU (%)")
        plt.title(f"CPU utilization over time: {r.name}")
        plt.tight_layout()
        plt.savefig(outdir / f"cpu_timeseries_{r.name}.png", dpi=160)
        plt.close()
        # Memory
        plt.figure()
        df.plot(x="ts", y="rss_mb", legend=False)
        plt.xlabel("timestamp (s)")
        plt.ylabel("RSS (MB)")
        plt.title(f"Memory (RSS) over time: {r.name}")
        plt.tight_layout()
        plt.savefig(outdir / f"memory_timeseries_{r.name}.png", dpi=160)
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Unified plots from summary.csv if present; fallback to detail.csv.")
    ap.add_argument("--csv", nargs="+", required=True, help="At least one CSV from each run (detail.csv or summary.csv).")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels matching the CSV order.")
    ap.add_argument("--out", default="plots", help="Output directory for images.")
    args = ap.parse_args()

    outdir = Path(args.out)
    ensure_out(outdir)

    runs: List[RunData] = []
    for i, p in enumerate(args.csv):
        label = args.labels[i] if args.labels and i < len(args.labels) else None
        runs.append(_load_run(Path(p), label))

    # Percentiles/mean bars (from summary if present)
    plot_bars(
        [r.name for r in runs],
        [r.mean_ms for r in runs],
        "Latency (ms)",
        "Mean latency",
        outdir / "mean_latency.png"
    )
    plot_bars(
        [r.name for r in runs],
        [r.p95_ms for r in runs],
        "Latency (ms)",
        "p95 latency",
        outdir / "p95_latency.png"
    )
    plot_bars(
        [r.name for r in runs],
        [r.p99_ms for r in runs],
        "Latency (ms)",
        "p99 latency",
        outdir / "p99_latency.png"
    )

    # Throughput + hit rate bars (from summary if present)
    plot_bars(
        [r.name for r in runs],
        [r.throughput_rps for r in runs],
        "Requests per second",
        "Throughput",
        outdir / "throughput.png"
    )
    plot_bars(
        [r.name for r in runs],
        [r.hit_rate_pct for r in runs],
        "Hit rate (%)",
        "Cache hit rate",
        outdir / "hit_rate.png"
    )

    # Per-run CPU/memory timeseries (auto)
    plot_cpu_mem(runs, outdir)

    print(f"Saved plots in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
