"""A multi-tier KV block cache with prefix-chained keys.

Blocks are identified the way paged-attention engines do it: the key of block
``i`` is a hash of the key of block ``i - 1`` and the tokens in block ``i``. A
block can therefore only be reused if its entire prefix is also reused, and a
prefix match is the longest run of leading blocks that are cached somewhere.

Each tier is an LRU over whole blocks. When a tier overflows, its victim is
demoted to the next tier; the last tier drops it. A hit in a lower tier
promotes the block back to the top tier, because the GPU needs it there.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .specs import TierSpec

BlockKey = int


def chain_keys(segments: Sequence[tuple[int, int]], block_tokens: int, parent: BlockKey = 0) -> list[BlockKey]:
    """Keys for a prompt made of ``(segment_id, n_blocks)`` pieces.

    Segments are block-aligned, a simplification that keeps the simulator fast
    while preserving the property that matters: every key commits to its full
    prefix, so identical text after different histories never collides.
    """
    del block_tokens  # segments are expressed in blocks already
    keys, key = [], parent
    for segment_id, n_blocks in segments:
        for index in range(n_blocks):
            key = hash((key, segment_id, index))
            keys.append(key)
    return keys


@dataclass
class Tier:
    spec: TierSpec
    block_bytes: int
    blocks: OrderedDict = field(default_factory=OrderedDict)

    @property
    def capacity_blocks(self) -> int:
        return self.spec.capacity_bytes // self.block_bytes

    def __contains__(self, key: BlockKey) -> bool:
        return key in self.blocks


@dataclass
class CacheStats:
    lookups: int = 0
    hit_blocks: dict[str, int] = field(default_factory=dict)
    miss_blocks: int = 0
    inserted_blocks: int = 0
    demotions: int = 0
    promotions: int = 0
    dropped_blocks: int = 0
    bytes_loaded: dict[str, int] = field(default_factory=dict)


class TieredKVCache:
    """An exclusive multi-tier LRU: every cached block lives in exactly one tier.

    ``policy`` controls the recency order *within* a prompt:

    ``lru``         blocks are touched front to back, so a chain's first block
                    is its least recently used and is evicted first. Losing an
                    interior block orphans every block after it: they stay
                    resident but can never be matched as a prefix again.
    ``leaf-first``  blocks are touched back to front (as vLLM does when it
                    returns a sequence's blocks to the free queue), so eviction
                    trims chains from the tail and never strands a suffix.
    """

    def __init__(self, tiers: Sequence[TierSpec], block_bytes: int, policy: str = "lru") -> None:
        if policy not in {"lru", "leaf-first"}:
            raise ValueError(f"unknown policy {policy!r}")
        if not tiers:
            raise ValueError("at least one tier is required")
        self.tiers = [Tier(spec, block_bytes) for spec in tiers]
        self.block_bytes = block_bytes
        self.policy = policy
        self.parent: dict[BlockKey, BlockKey] = {}
        self.stats = CacheStats(
            hit_blocks={t.spec.name: 0 for t in self.tiers},
            bytes_loaded={t.spec.name: 0 for t in self.tiers},
        )

    # ------------------------------------------------------------------ queries
    def locate(self, key: BlockKey) -> int | None:
        for index, tier in enumerate(self.tiers):
            if key in tier:
                return index
        return None

    def match_prefix(self, keys: Sequence[BlockKey]) -> list[int]:
        """Tier index of each leading cached block, stopping at the first miss."""
        found = []
        for key in keys:
            where = self.locate(key)
            if where is None:
                break
            found.append(where)
        return found

    def orphaned_blocks(self) -> int:
        """Resident blocks with a missing ancestor: capacity that can never produce a hit."""
        reachable: dict[BlockKey, bool] = {0: True}

        def is_reachable(key: BlockKey) -> bool:
            chain = []
            while key not in reachable:
                if self.locate(key) is None:
                    reachable[key] = False
                    break
                chain.append(key)
                key = self.parent.get(key, 0)
            verdict = reachable[key]
            for k in chain:
                reachable[k] = verdict
            return verdict

        return sum(1 for tier in self.tiers for key in tier.blocks if not is_reachable(key))

    def resident_blocks(self) -> dict[str, int]:
        return {t.spec.name: len(t.blocks) for t in self.tiers}

    # --------------------------------------------------------------- mutations
    def access(self, keys: Sequence[BlockKey], *, record: bool = True) -> list[int]:
        """Serve a prompt: count prefix hits per tier, then make every block resident on top.

        Returns the tier each matched prefix block was found in. With
        ``record=False`` blocks are admitted without touching hit statistics,
        which is how freshly decoded reply blocks enter the cache.
        """
        matched = self.match_prefix(keys)
        if record:
            self.stats.lookups += 1
            for where in matched:
                name = self.tiers[where].spec.name
                self.stats.hit_blocks[name] += 1
                self.stats.bytes_loaded[name] += self.block_bytes if where > 0 else 0
            self.stats.miss_blocks += len(keys) - len(matched)
        previous = 0
        for key in keys:
            self.parent.setdefault(key, previous)
            previous = key
        order = reversed(keys) if self.policy == "leaf-first" else keys
        top = self.tiers[0].blocks
        for key in order:
            # Re-locate every time: inserting earlier blocks may have demoted this one.
            where = self.locate(key)
            if where == 0:
                top.move_to_end(key)
                continue
            if where is None:
                self.stats.inserted_blocks += 1
            else:
                del self.tiers[where].blocks[key]
                self.stats.promotions += 1
            self._insert(0, key)
        return matched

    def _insert(self, level: int, key: BlockKey) -> None:
        tier = self.tiers[level]
        tier.blocks[key] = None
        tier.blocks.move_to_end(key)
        while len(tier.blocks) > tier.capacity_blocks:
            victim = next(iter(tier.blocks))
            if victim == key:  # a single block larger than the tier: cannot cache it here
                victim = key
            del tier.blocks[victim]
            if level + 1 < len(self.tiers):
                self.stats.demotions += 1
                self._insert(level + 1, victim)
            else:
                self.stats.dropped_blocks += 1
