
"""
Lets the eviction policy look up cost by id, without passing functions through the factory. 
The benchmark sets this hook at runtime
"""
from typing import Callable

_provider: Callable[[int], float] | None = None

def set_provider(fn: Callable[[int], float]) -> None:
    global _provider
    _provider = fn

def get_cost(cache_id: int) -> float:
    if _provider is None:
        # Neutral default (never prefer to evict if unknown, treat it as "high")
        return float("inf")
    try:
        return float(_provider(int(cache_id)))
    except Exception:
        return float("inf")
