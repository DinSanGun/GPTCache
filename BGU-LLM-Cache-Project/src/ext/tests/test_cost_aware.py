import pytest

from gptcache.manager.eviction.memory_cache import MemoryCacheEviction
from gptcache.manager.eviction import cost_provider, cost_inbox


def drain_inbox():
    # Helper to make tests independent: empty the inbox between steps
    while cost_inbox.pop_last() is not None:
        pass


def test_min_cost_eviction_order():
    """
    Given explicit costs, on overflow the COST_AWARE policy must evict the cheapest id.
    """
    drain_inbox()
    costs = {10: 1.0, 11: 5.0, 12: 0.5, 13: 9.0}
    cost_provider.set_provider(lambda k: costs.get(int(k), 1e9))

    evicted = []
    ev = MemoryCacheEviction(
        policy="COST_AWARE", maxsize=3, clean_size=1, on_evict=lambda keys: evicted.extend(keys)
    )

    ev.put([10, 11, 12])   # fill
    ev.put([13])           # overflow -> should evict id=12 (cost 0.5)
    assert evicted[:1] == [12]


def test_getsizeof_is_forwarded_and_respected():
    """
    If a getsizeof is provided, capacity is counted accordingly,
    but the victim is still the *cheapest* (min-cost) key.
    """
    drain_inbox()
    costs = {1: 10.0, 2: 20.0, 3: 30.0}
    cost_provider.set_provider(lambda k: costs.get(int(k), 1e9))

    evicted = []
    # Each value "weighs" 5, so maxsize=10 can hold 2 items; inserting the 3rd triggers 1 eviction.
    ev = MemoryCacheEviction(
        policy="COST_AWARE", maxsize=10, clean_size=1,
        on_evict=lambda keys: evicted.extend(keys),
        getsizeof=lambda v: 5,
    )

    ev.put([1, 2])   # size = 10
    ev.put([3])      # overflow -> evict cheapest among {1(10), 2(20), 3(30)} => 1
    assert evicted == [1]


def test_inbox_pushes_inserted_ids_in_order():
    """
    MemoryCacheEviction.put must push inserted ids to the inbox so the bench can grab them
    without hitting the DB.
    """
    drain_inbox()
    cost_provider.set_provider(lambda k: 1.0)  # costs don't matter here

    _ = MemoryCacheEviction(policy="COST_AWARE", maxsize=3, clean_size=1, on_evict=lambda k: None)
    _.put([100, 101, 102])

    # Inbox pops the last pushed id first
    assert cost_inbox.pop_last() == 102
    assert cost_inbox.pop_last() == 101
    assert cost_inbox.pop_last() == 100
    assert cost_inbox.pop_last() is None   # now empty


def test_clean_size_controls_how_many_evictions_per_overflow():
    """
    clean_size=2 should evict two cheapest items when a single insertion overflows the cache.
    """
    drain_inbox()
    costs = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    cost_provider.set_provider(lambda k: costs.get(int(k), 1e9))

    evicted = []
    ev = MemoryCacheEviction(
        policy="COST_AWARE", maxsize=3, clean_size=2, on_evict=lambda keys: evicted.extend(keys)
    )

    ev.put([1, 2, 3])  # fill
    ev.put([4])        # overflow -> should evict 2 cheapest of {1,2,3} => [1,2]
    assert evicted[:2] == [1, 2]


def test_unknown_costs_are_treated_as_infinite_and_kept():
    """
    If a key has unknown cost, provider returns +inf. COST_AWARE should prefer evicting
    known (finite) cheaper keys first, keeping unknowns.
    """
    drain_inbox()
    costs = {10: 1.0, 11: 2.0}  # 42 is unknown -> +inf
    cost_provider.set_provider(lambda k: costs.get(int(k), float("inf")))

    evicted = []
    ev = MemoryCacheEviction(
        policy="COST_AWARE", maxsize=2, clean_size=1, on_evict=lambda keys: evicted.extend(keys)
    )

    ev.put([10])     # present: {10(1)}
    ev.put([42])     # present: {10(1), 42(inf)}
    ev.put([11])     # overflow: victim should be 10 (min cost), not 42(inf)
    assert evicted[:1] == [10]
