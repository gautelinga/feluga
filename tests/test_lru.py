from feluga.lru import LRU


def test_keeps_the_most_recently_used():
    cache = LRU(2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.get("a") == 1          # a is now the most recently used
    cache["c"] = 3
    assert list(cache) == ["a", "c"]
    assert cache.get("b") is None
