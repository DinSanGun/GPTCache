from typing import Any, Callable, List, Optional, Iterator

import cachetools

from gptcache.manager.eviction.base import EvictionBase
from gptcache.manager.eviction import cost_provider


def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        wrapper_func(keys)

    return wrapper


class MemoryCacheEviction(EvictionBase):
    """eviction: Memory Cache

    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]


    """

    def __init__(
            self,
            policy: str = "LRU",
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs,
    ):
        self._policy = policy.upper()
        if self._policy == "LRU":
            self._cache = cachetools.LRUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "LFU":
            self._cache = cachetools.LFUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "FIFO":
            self._cache = cachetools.FIFOCache(maxsize=maxsize, **kwargs)
        elif self._policy == "RR":
            self._cache = cachetools.RRCache(maxsize=maxsize, **kwargs)
        elif self._policy == "COST_AWARE":
            self._cache = CostAwareCache(maxsize=maxsize)
        else:
            raise ValueError(f"Unknown policy {policy}")

        self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def put(self, objs: List[Any]):
        for obj in objs:
            self._cache[obj] = True

    def get(self, obj: Any):
        return self._cache.get(obj)

    @property
    def policy(self) -> str:
        return self._policy
    
class CostAwareCache(cachetools.Cache):
    """
    A tiny cachetools Cache variant that evicts the *cheapest to recompute* ids.
    Stores only keys (GPTCache internal ids). Value is ignored (always True).
    On overflow, popitem() returns the id with MIN(get_cost(id)).
    """
    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        # cachetools will call popitem() while len(self) > maxsize

    def popitem(self) -> tuple[Any, Any]:
        if not self._Cache__data:  # type: ignore[attr-defined]
            raise KeyError("%s is empty" % type(self).__name__)
        # Choose the id with minimal cost
        min_key: Optional[Any] = None
        min_cost: float = float("inf")
        # Iterate over current keys (ids)
        for k in self._Cache__data.keys():  # type: ignore[attr-defined]
            c = cost_provider.get_cost(k)
            if c < min_cost:
                min_cost, min_key = c, k
        # Fallback: arbitrary pop if everything is inf/None
        if min_key is None:
            return self._Cache__data.popitem()  # type: ignore[attr-defined]
        v = self._Cache__data.pop(min_key)     # type: ignore[attr-defined]
        return (min_key, v)
