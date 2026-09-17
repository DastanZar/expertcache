import pytest
import torch

from expertcache.core import Entry, Residency
from expertcache.execute import Engine, make_bank, sizes_of


def test_cost_density_and_aging():
    cache = Residency({0: 4, 1: 16}, 20, "cost2q")
    cache.resident = {0: Entry(4, 1, 2), 1: Entry(16, 1, 2)}
    cache.clock = 3
    assert cache.eviction_rank(0) > cache.eviction_rank(1)
    assert cache.eviction_rank(0)[0] == 1
    cache.clock = 40
    assert cache.eviction_rank(0)[0] == 0


def test_execution_exception_releases_pins():
    bank = make_bank([8, 12], width=8)
    engine = Engine(bank, sum(sizes_of(bank).values()))
    with pytest.raises(RuntimeError):
        engine.forward(torch.ones(1, 9), [0, 1], [0.5, 0.5])
    assert not engine.cache.pinned
    engine.cache.invariant()
    assert torch.isfinite(engine.forward(torch.ones(1, 8), [0, 1], [0.5, 0.5])).all()
