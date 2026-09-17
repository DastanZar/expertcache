import random

import pytest

from expertcache.core import Residency


@pytest.mark.parametrize("policy", ["lru", "cost2q"])
def test_random_variable_size_transaction_invariants(policy):
    rng = random.Random(42)
    sizes = {i: rng.randrange(1, 20) for i in range(15)}
    cache = Residency(sizes, 50, policy)
    for _ in range(400):
        route = rng.sample(list(sizes), 2)
        before_bytes = cache.bytes_loaded
        before_missing = set(route) - cache.resident.keys()
        with cache.lease(route):
            assert cache.pinned == set(route)
            cache.invariant()
        assert cache.bytes_loaded - before_bytes == sum(sizes[k] for k in before_missing)
        assert cache.pins == {}


def test_unknown_key_does_not_mutate():
    cache = Residency({0: 2}, 3)
    with pytest.raises(KeyError), cache.lease([0, 99]):
        pass
    assert cache.misses == 0 and not cache.resident


@pytest.mark.parametrize(
    "sizes,capacity,policy", [({0: 0}, 2, "lru"), ({0: 1}, 0, "lru"), ({0: 1}, 3, "oracle")]
)
def test_invalid_configuration(sizes, capacity, policy):
    with pytest.raises(ValueError):
        Residency(sizes, capacity, policy)
