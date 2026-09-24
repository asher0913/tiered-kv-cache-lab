import pytest

from kvtier.cache import TieredKVCache, chain_keys
from kvtier.specs import MODELS, TierSpec

BLOCK = 1024


def tier(name: str, blocks: int, bandwidth: float = 1e9) -> TierSpec:
    return TierSpec(name, blocks * BLOCK, bandwidth)


def test_model_kv_footprint_matches_published_configs():
    assert MODELS["llama-3-8b"].kv_bytes_per_token == 131_072  # 128 KiB
    assert MODELS["llama-3-70b"].kv_bytes_per_token == 327_680  # 320 KiB
    assert MODELS["qwen2.5-7b"].kv_bytes_per_token == 57_344  # 56 KiB


def test_transfer_time_is_latency_plus_bytes_over_bandwidth():
    spec = TierSpec("dram", 10**12, 25e9, 10e-6)
    assert spec.transfer_seconds(25 * 10**9) == pytest.approx(1.0 + 10e-6)
    assert spec.transfer_seconds(0) == 0.0


def test_keys_commit_to_the_whole_prefix():
    a = chain_keys([(1, 3), (7, 2)], 16)
    b = chain_keys([(2, 3), (7, 2)], 16)
    assert len(a) == 5
    assert a[:3] != b[:3]
    assert a[3:] != b[3:]  # same segment after a different history is a different block
    assert chain_keys([(1, 3)], 16) == a[:3]
    assert chain_keys([(7, 2)], 16, parent=a[2]) == a[3:]


def test_prefix_match_stops_at_first_miss():
    cache = TieredKVCache([tier("hbm", 100)], BLOCK)
    keys = chain_keys([(1, 6)], 16)
    cache.access(keys[:4])
    assert cache.match_prefix(keys) == [0, 0, 0, 0]
    assert cache.match_prefix([keys[0], 999, keys[2]]) == [0]


def test_tiers_are_exclusive_and_capacity_is_respected():
    cache = TieredKVCache([tier("hbm", 4), tier("dram", 6)], BLOCK)
    for session in range(5):
        cache.access(chain_keys([(session, 3)], 16))
    resident = cache.resident_blocks()
    assert resident["hbm"] <= 4 and resident["dram"] <= 6
    hbm, dram = set(cache.tiers[0].blocks), set(cache.tiers[1].blocks)
    assert not hbm & dram


def test_overflow_demotes_and_hit_promotes():
    cache = TieredKVCache([tier("hbm", 3), tier("dram", 10)], BLOCK)
    first = chain_keys([(1, 3)], 16)
    cache.access(first)
    cache.access(chain_keys([(2, 3)], 16))  # pushes the first chain down
    assert all(cache.locate(k) == 1 for k in first)
    matched = cache.access(first)
    assert matched == [1, 1, 1]
    assert all(cache.locate(k) == 0 for k in first)
    assert cache.stats.promotions == 3 and cache.stats.demotions >= 3


def test_last_tier_drops():
    cache = TieredKVCache([tier("hbm", 2)], BLOCK)
    cache.access(chain_keys([(1, 2)], 16))
    cache.access(chain_keys([(2, 2)], 16))
    assert cache.stats.dropped_blocks == 2
    assert cache.match_prefix(chain_keys([(1, 2)], 16)) == []


def test_leaf_first_trims_chains_from_the_tail():
    old = chain_keys([(1, 4)], 16)
    lru = TieredKVCache([tier("hbm", 5)], BLOCK, policy="lru")
    leaf = TieredKVCache([tier("hbm", 5)], BLOCK, policy="leaf-first")
    for cache in (lru, leaf):
        cache.access(old)
        cache.access(chain_keys([(2, 2)], 16))  # one block must go
    # LRU evicted the head, so the three surviving blocks are unreachable
    assert lru.match_prefix(old) == []
    assert lru.orphaned_blocks() == 3
    # leaf-first evicted the tail, so the first three blocks still hit
    assert leaf.match_prefix(old) == [0, 0, 0]
    assert leaf.orphaned_blocks() == 0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        TieredKVCache([], BLOCK)
    with pytest.raises(ValueError):
        TieredKVCache([tier("hbm", 1)], BLOCK, policy="random")
