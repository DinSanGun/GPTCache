from gptcache.manager.eviction.memory_cache import MemoryCacheEviction
from gptcache.manager.eviction import cost_provider
cost_provider.set_provider(lambda k: {10:1.0, 11:5.0, 12:0.5, 13:9.0, 14:7.0}.get(int(k), 1e9))
evicted = []
def on_evict(keys): evicted.extend(keys)

ev = MemoryCacheEviction(policy="COST_AWARE", maxsize=4, clean_size=1, on_evict=on_evict, getsizeof=lambda v: 1)
ev.put([10,11,12,13])  # fill
ev.put([14])           # should evict 12 (cheapest)
print("evicted:", evicted)   # expect starts with [12]
