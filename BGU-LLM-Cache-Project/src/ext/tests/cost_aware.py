from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Callable, Tuple
from gptcache import cache as GLOBAL_CACHE
from gptcache.manager.eviction import cost_provider

Metric = Literal["latency_ms", "tokens"]

@dataclass
class _Config:
    metric: Metric = "latency_ms"
    decay: float = 0.0  # EMA weight for new observations

_cfg = _Config()
_id_to_cost: Dict[int, float] = {}
_prompt_to_id: Dict[str, int] = {}

def install_cost_provider(metric: Metric = "latency_ms", decay: float = 0.0) -> None:
    _cfg.metric = metric
    _cfg.decay  = max(0.0, min(1.0, float(decay)))
    cost_provider.set_provider(lambda cid: _id_to_cost.get(int(cid), float("inf")))

def get_live_ids() -> set[int]:
    # All non-deleted scalar rows (SQLite ids)
    return set(GLOBAL_CACHE.data_manager.s.get_ids(deleted=False))

def _ema(old: Optional[float], new: float) -> float:
    if old is None: return new
    d = _cfg.decay
    return (1.0 - d) * old + d * new

def record_mapping(prompt: str, new_id: int) -> None:
    _prompt_to_id[prompt] = int(new_id)

def record_cost_by_id(cache_id: int, observed_cost: float) -> None:
    cid = int(cache_id)
    _id_to_cost[cid] = _ema(_id_to_cost.get(cid), float(observed_cost))

def cost_for_prompt(prompt: str) -> float:
    cid = _prompt_to_id.get(prompt)
    if cid is None: 
        return 0.0
    return float(_id_to_cost.get(cid, 0.0))

def discover_new_id(before: set[int], after: set[int]) -> Optional[int]:
    delta = list(after - before)
    return int(delta[0]) if delta else None
