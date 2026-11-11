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
