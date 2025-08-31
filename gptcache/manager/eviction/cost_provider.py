"""
Global hook so eviction policy can ask: cost for this cache-id?
Your bench sets this at runtime with set_provider function
"""
from typing import Callable, Optional

_Provider: Optional[Callable[[int], float]] = None

def set_provider(fn: Callable[[int], float]) -> None:
    global _Provider
    _Provider = fn

def get_cost(cache_id: int) -> float:
    if _Provider is None:
        return float("inf")
    try:
        return float(_Provider(int(cache_id)))
    except Exception:
        return float("inf")
