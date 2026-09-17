"""CPU module offload emulation with actual independent parameter copies."""

import copy
import math

import torch
from torch import Tensor, nn

from .core import Residency

Bank = dict[int, nn.Module]


def make_bank(hidden: list[int], width: int = 16, seed: int = 19) -> Bank:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return {
            i: nn.Sequential(nn.Linear(width, h), nn.GELU(), nn.Linear(h, width)).eval()
            for i, h in enumerate(hidden)
        }


def sizes_of(bank: Bank) -> dict[int, int]:
    return {
        key: sum(p.numel() * p.element_size() for p in module.parameters())
        for key, module in bank.items()
    }


def validate(route: list[int], weights: list[float]) -> None:
    if not route or len(route) != len(weights):
        raise ValueError("nonempty route and aligned weights required")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("finite nonnegative weights required")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("weights must sum to one")


@torch.no_grad()
def resident_forward(bank: Bank, x: Tensor, route: list[int], weights: list[float]) -> Tensor:
    validate(route, weights)
    result = torch.zeros_like(x)
    for key, weight in zip(route, weights, strict=True):
        result = result + weight * bank[key](x)
    return result


class Engine:
    def __init__(self, bank: Bank, capacity: int, policy: str = "lru") -> None:
        self.bank = bank
        self.cache = Residency(sizes_of(bank), capacity, policy)
        self.loaded: Bank = {}
        self.actual_bytes_copied = 0

    @torch.no_grad()
    def forward(self, x: Tensor, route: list[int], weights: list[float]) -> Tensor:
        validate(route, weights)
        with self.cache.lease(route):
            # Delete before copying: loaded module payload never exceeds the byte budget.
            for key in list(self.loaded):
                if key not in self.cache.resident:
                    del self.loaded[key]
            try:
                for key in route:
                    if key not in self.loaded:
                        self.loaded[key] = copy.deepcopy(self.bank[key])
                        self.actual_bytes_copied += self.cache.sizes[key]
            except Exception:
                # Failed materialization cannot leave a phantom resident hit.
                for key in list(self.cache.resident):
                    if key not in self.loaded:
                        del self.cache.resident[key]
                # lease's finally releases pins before checking its invariant.
                raise
            return resident_forward(self.loaded, x, route, weights)
