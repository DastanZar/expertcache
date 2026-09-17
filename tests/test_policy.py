import copy

import pytest

from expertcache.core import Residency


def test_cost2q_resists_one_pass_scan():
    caches = [Residency(dict.fromkeys(range(20), 4), 12, policy=p) for p in ["lru", "cost2q"]]
    for cache in caches:
        for key in [0, 1, 0, 1] + list(range(2, 20)) + [0, 1]:
            with cache.lease([key]):
                cache.invariant()
    assert caches[1].misses < caches[0].misses


def test_atomic_capacity_failure_and_nested_pin_release():
    cache = Residency({0: 4, 1: 7, 2: 6}, 11)
    with cache.lease([0]):
        before = copy.deepcopy(cache.__dict__)
        with pytest.raises(MemoryError), cache.lease([1, 2]):
            pass
        assert cache.__dict__ == before
        with cache.lease([0, 1, 1]):
            assert cache.pins == {0: 2, 1: 1}
        assert cache.pins == {0: 1}
    with pytest.raises(RuntimeError), cache.lease([2]):
        raise RuntimeError("execution failed")
    assert cache.pins == {}
    cache.invariant()
