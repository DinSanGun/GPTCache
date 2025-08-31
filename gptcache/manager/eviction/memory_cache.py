from typing import Any, Callable, List, Optional, Iterator

import cachetools
from gptcache.manager.eviction.base import EvictionBase

import heapq
from gptcache.manager.eviction import cost_provider, cost_inbox


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
            self._cache = CostAwareCache(maxsize=maxsize, **kwargs)
        else:
            raise ValueError(f"Unknown policy {policy}")

        self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def put(self, objs: List[Any]):
        for obj in objs:
            self._cache[obj] = True
            # tell the bench which id was just inserted (no DB round-trip)
            try:
                cost_inbox.push(obj)
            except Exception:
                pass


    def get(self, obj: Any):
        return self._cache.get(obj)

    @property
    def policy(self) -> str:
        return self._policy

class CostAwareCache(cachetools.Cache):
    """
    Evicts the cheapest to recompute key using a min-heap of (cost, key).
    Values are unused (stored as True); we still keep cachetools' size accounting.
    """
    def __init__(self, maxsize: int, **kwargs):
        super().__init__(maxsize=maxsize, getsizeof=kwargs.get("getsizeof"))
        self._heap = []  # list[(float, Any)]

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        heapq.heappush(self._heap, (float(cost_provider.get_cost(key)), key))
        # cachetools will call self.popitem() while currsize > maxsize

    def popitem(self):
        if not self._Cache__data:  # type: ignore[attr-defined]
            raise KeyError("%s is empty" % type(self).__name__)
        # pull until we find a live key
        while self._heap:
            _, k = heapq.heappop(self._heap)
            if k in self._Cache__data:  # type: ignore[attr-defined]
                v = self._Cache__data.pop(k)  # type: ignore[attr-defined]
                # keep cachetools' size accounting correct
                getsizeof = getattr(self, "_Cache__getsizeof", None)
                if callable(getsizeof):
                    try:
                        self._Cache__currsize -= max(1, int(getsizeof(v)))
                    except Exception:
                        self._Cache__currsize = len(self._Cache__data) 
                else:
                    self._Cache__currsize = len(self._Cache__data)

                return (k, v)
        # fallback if heap got out of sync
        k, v = self._Cache__data.popitem()  # type: ignore[attr-defined]
        getsizeof = getattr(self, "_Cache__getsizeof", None)
        if callable(getsizeof):
            try:
                self._Cache__currsize -= max(1, int(getsizeof(v))) 
            except Exception:
                self._Cache__currsize = len(self._Cache__data)
        else:
            self._Cache__currsize = len(self._Cache__data)

        return (k, v)
