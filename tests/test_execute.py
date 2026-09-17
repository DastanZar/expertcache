import pytest
import torch


def test_real_experts_parity_and_reuse():
    from expertcache.execute import Engine, make_bank, resident_forward, sizes_of

    bank = make_bank([8, 12, 16, 20], width=8)
    sizes = sizes_of(bank)
    engine = Engine(bank, sum(sorted(sizes.values())[-2:]), "cost2q")
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8) / 8
    for route in [[0, 1], [2, 3], [0, 3], [3, 3], [0, 1]]:
        weights = [0.3, 0.7]
        result = engine.forward(x, route, weights)
        expected = resident_forward(bank, x, route, weights)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert set(engine.loaded) == set(engine.cache.resident)
        assert sum(sizes[k] for k in engine.loaded) <= engine.cache.capacity
        assert not engine.cache.pinned
    assert engine.cache.bytes_loaded == engine.actual_bytes_copied
    for key, module in engine.loaded.items():
        for a, b in zip(module.parameters(), bank[key].parameters(), strict=True):
            assert a.data_ptr() != b.data_ptr()


def test_rejects_invalid_route_before_loading():
    from expertcache.execute import Engine, make_bank, sizes_of

    bank = make_bank([8, 12], width=8)
    engine = Engine(bank, max(sizes_of(bank).values()), "lru")
    with pytest.raises(MemoryError):
        engine.forward(torch.ones(1, 8), [0, 1], [0.5, 0.5])
    assert engine.cache.misses == 0 and not engine.loaded
    with pytest.raises(ValueError):
        engine.forward(torch.ones(1, 8), [0], [float("nan")])
