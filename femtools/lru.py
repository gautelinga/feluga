"""A dict that keeps only its most recently used entries, for the caches of
objects built per space, form or condition: a program that keeps making new ones
cycles through the cache instead of filling memory."""
from collections import OrderedDict


class LRU(OrderedDict):
    """At most `capacity` entries. `get` and setting an entry make it the most
    recently used; setting one beyond the capacity drops the least recently used."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def get(self, key, default=None):
        if key not in self:
            return default
        self.move_to_end(key)
        return self[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)


class WeakCache:
    """Values computed once per live object (a space, a UniqueDofs), dropped with it: the
    object is not kept alive, and a later object reusing its id is not served."""

    def __init__(self):
        self._d = {}

    def get(self, objs, build):
        """The value for the tuple of objects `objs`, `build()` the first time."""
        import weakref
        key = tuple(id(o) for o in objs)
        hit = self._d.get(key)
        if hit is not None and all(r() is o for r, o in zip(hit[0], objs)):
            return hit[1]
        value = build()
        refs = tuple(weakref.ref(o) for o in objs)
        self._d[key] = (refs, value)
        for o in objs:
            weakref.finalize(o, self._d.pop, key, None)
        return value

    def __len__(self):
        return len(self._d)
