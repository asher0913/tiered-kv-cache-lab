import unittest

from tiered_kv_cache_lab.core import TierSpec, TieredKVCache, little_law_capacity


class CacheTests(unittest.TestCase):
    def cache(self):
        return TieredKVCache([TierSpec("hbm", 8, 0.1), TierSpec("dram", 16, 1.0), TierSpec("nvme", 32, 8.0)])

    def test_eviction_demotes_and_hit_promotes(self):
        cache = self.cache()
        cache.put("a", 8)
        cache.put("b", 8)
        self.assertIn("a", cache.data[1])
        self.assertEqual(cache.get("a").key, "a")
        self.assertIn("a", cache.data[0])
        self.assertGreaterEqual(cache.hits["dram"], 1)

    def test_capacity_and_accounting(self):
        cache = self.cache()
        for i in range(12):
            cache.put(str(i), 8)
        self.assertTrue(all(used <= spec.capacity for used, spec in zip(cache.used, cache.specs)))
        self.assertGreater(cache.drops, 0)

    def test_request_admits_misses(self):
        cache = self.cache()
        cache.request(["x", "x"])
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits["hbm"], 1)

    def test_little_law(self):
        self.assertEqual(little_law_capacity(100, 20, 0.5), 4)


if __name__ == "__main__":
    unittest.main()
