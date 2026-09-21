from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
import random
from statistics import median


@dataclass(frozen=True)
class TierSpec:
    name: str
    capacity: int
    latency_ms: float


@dataclass(frozen=True)
class Block:
    key: str
    size: int


class TieredKVCache:
    def __init__(self, specs: list[TierSpec]) -> None:
        if not specs or any(s.capacity <= 0 or s.latency_ms < 0 for s in specs):
            raise ValueError("tiers require positive capacity and non-negative latency")
        self.specs = specs
        self.data = [OrderedDict() for _ in specs]
        self.used = [0 for _ in specs]
        self.hits = {s.name: 0 for s in specs}
        self.misses = 0
        self.evictions = 0
        self.drops = 0
        self.latencies: list[float] = []

    def _remove(self, tier: int, key: str) -> Block:
        block = self.data[tier].pop(key)
        self.used[tier] -= block.size
        return block

    def _insert(self, tier: int, block: Block) -> None:
        if block.size > self.specs[tier].capacity:
            if tier + 1 < len(self.specs):
                self._insert(tier + 1, block)
            else:
                self.drops += 1
            return
        if block.key in self.data[tier]:
            self._remove(tier, block.key)
        while self.used[tier] + block.size > self.specs[tier].capacity:
            _, victim = self.data[tier].popitem(last=False)
            self.used[tier] -= victim.size
            self.evictions += 1
            if tier + 1 < len(self.specs):
                self._insert(tier + 1, victim)
            else:
                self.drops += 1
        self.data[tier][block.key] = block
        self.used[tier] += block.size

    def put(self, key: str, size: int) -> None:
        if size <= 0:
            raise ValueError("block size must be positive")
        for index, tier in enumerate(self.data):
            if key in tier:
                self._remove(index, key)
        self._insert(0, Block(key, size))

    def get(self, key: str) -> Block | None:
        for index, tier in enumerate(self.data):
            if key not in tier:
                continue
            block = self._remove(index, key)
            self.hits[self.specs[index].name] += 1
            latency = self.specs[index].latency_ms
            self.latencies.append(latency)
            self._insert(0, block)
            return block
        self.misses += 1
        self.latencies.append(self.specs[-1].latency_ms * 4)
        return None

    def request(self, prefixes: list[str], block_size: int = 8) -> float:
        before = len(self.latencies)
        for prefix in prefixes:
            if self.get(prefix) is None:
                self.put(prefix, block_size)
        return sum(self.latencies[before:])

    def report(self) -> dict[str, object]:
        ordered = sorted(self.latencies)
        percentile = lambda q: ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0
        return {
            "hits": dict(self.hits),
            "misses": self.misses,
            "evictions": self.evictions,
            "drops": self.drops,
            "used": dict(zip((s.name for s in self.specs), self.used)),
            "p50_lookup_ms": median(ordered) if ordered else 0.0,
            "p95_lookup_ms": percentile(0.95),
        }


def little_law_capacity(arrival_rps: float, service_ms: float, utilization: float = 0.8) -> int:
    if arrival_rps < 0 or service_ms < 0 or not 0 < utilization <= 1:
        raise ValueError("invalid capacity inputs")
    return math.ceil(arrival_rps * service_ms / 1000 / utilization)


def demo(seed: int = 7) -> dict[str, object]:
    rng = random.Random(seed)
    cache = TieredKVCache([
        TierSpec("hbm", 64, 0.08),
        TierSpec("dram", 192, 0.8),
        TierSpec("nvme", 512, 6.5),
    ])
    keys = [f"prefix-{i}" for i in range(80)]
    weights = [1 / (i + 1) ** 1.15 for i in range(len(keys))]
    for _ in range(240):
        chosen = rng.choices(keys, weights=weights, k=rng.randint(1, 4))
        cache.request(chosen)
    report = cache.report()
    report["recommended_concurrency"] = little_law_capacity(18, float(report["p95_lookup_ms"]))
    return report


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
