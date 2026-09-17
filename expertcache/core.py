"""Online byte-budget residency; decisions never inspect future requests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Entry:
    size: int
    last: int
    hits: int = 1


class Residency:
    def __init__(self, sizes: dict[int, int], capacity: int, policy: str = "lru") -> None:
        if capacity <= 0 or any(size <= 0 for size in sizes.values()):
            raise ValueError("positive sizes and capacity required")
        if policy not in {"lru", "cost2q"}:
            raise ValueError("unknown policy")
        self.sizes = dict(sizes)
        self.capacity = capacity
        self.policy = policy
        self.resident: dict[int, Entry] = {}
        self.pins: dict[int, int] = {}
        self.clock = 0
        self.misses = 0
        self.hits = 0
        self.bytes_loaded = 0
        self.evictions = 0

    @property
    def used(self) -> int:
        return sum(entry.size for entry in self.resident.values())

    @property
    def pinned(self) -> set[int]:
        return {key for key, count in self.pins.items() if count}

    def invariant(self) -> None:
        assert self.used <= self.capacity
        assert self.pinned <= self.resident.keys()
        assert all(count > 0 for count in self.pins.values())

    def eviction_rank(self, key: int) -> tuple[float, ...]:
        entry = self.resident[key]
        if self.policy == "lru":
            return (float(entry.last),)
        # Two-hit promotion protects reuse from scans; aging prevents immortal hot entries.
        age = self.clock - entry.last + 1
        protected = entry.hits >= 2 and age <= 32
        # Fixed 4KiB setup-equivalent cost + payload: modeled, not timed PCIe.
        value_density = min(entry.hits, 8) * (4096 + entry.size) / (entry.size * age)
        return (float(protected), value_density, float(entry.last))

    @contextmanager
    def lease(self, keys: list[int]) -> Iterator[None]:
        requested = list(dict.fromkeys(keys))
        if any(key not in self.sizes for key in requested):
            raise KeyError("unknown expert")
        required = self.pinned | set(requested)
        if sum(self.sizes[key] for key in required) > self.capacity:
            raise MemoryError("pinned union exceeds capacity")
        for key in requested:
            self.clock += 1
            if key in self.resident:
                self.hits += 1
                self.resident[key].last = self.clock
                self.resident[key].hits += 1
            else:
                while self.used + self.sizes[key] > self.capacity:
                    candidates = [k for k in self.resident if k not in required]
                    victim = min(candidates, key=self.eviction_rank)
                    del self.resident[victim]
                    self.evictions += 1
                self.resident[key] = Entry(self.sizes[key], self.clock)
                self.misses += 1
                self.bytes_loaded += self.sizes[key]
            self.pins[key] = self.pins.get(key, 0) + 1
        self.invariant()
        try:
            yield
        finally:
            for key in requested:
                self.pins[key] -= 1
                if not self.pins[key]:
                    del self.pins[key]
            self.invariant()
