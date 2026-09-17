import expertcache.core as core


def test_byte_capacity():
    cache = core.Residency({0: 4, 1: 7, 2: 6}, 11, policy="lru")
    with cache.lease([0, 1]):
        assert cache.used == 11
        assert cache.pinned == {0, 1}
    with cache.lease([2]):
        assert cache.used <= 11
        assert 2 in cache.resident
    assert not cache.pinned
