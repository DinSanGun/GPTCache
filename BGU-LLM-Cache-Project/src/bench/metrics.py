from __future__ import annotations
import time, threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import psutil
import pandas as pd

@dataclass
class Sample:
    ts: float
    cpu_pct: float
    rss_mb: float

class SysSampler:
    def __init__(self, interval_ms: int = 300):
        self.interval = interval_ms / 1000.0
        self.samples: List[Sample] = []
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self):
        if self._thr: return
        proc = psutil.Process()
        psutil.cpu_percent(None)  # prime
        def loop():
            while not self._stop.is_set():
                cpu = psutil.cpu_percent(None)
                rss = proc.memory_info().rss / (1024*1024)
                self.samples.append(Sample(time.time(), cpu, rss))
                time.sleep(self.interval)
        self._thr = threading.Thread(target=loop, daemon=True)
        self._thr.start()

    def stop(self):
        if not self._thr: return
        self._stop.set()
        self._thr.join(timeout=2)

    def summarize(self) -> Dict[str, Any]:
        if not self.samples:
            return {"cpu_pct_mean": 0, "cpu_pct_max": 0, "rss_mb_peak": 0}
        cpus = [s.cpu_pct for s in self.samples]
        rsss = [s.rss_mb for s in self.samples]
        return {
            "cpu_pct_mean": sum(cpus)/len(cpus),
            "cpu_pct_max": max(cpus),
            "rss_mb_peak": max(rsss),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{"ts": s.ts, "cpu_pct": s.cpu_pct, "rss_mb": s.rss_mb} for s in self.samples])

def percentiles(values, ps=(0.95, 0.99)):
    if not values: 
        return {f"p{int(p*100)}": 0 for p in ps}
    s = sorted(values)
    out = {}
    for p in ps:
        k = int((len(s)-1)*p)
        out[f"p{int(p*100)}"] = s[k]
    return out
