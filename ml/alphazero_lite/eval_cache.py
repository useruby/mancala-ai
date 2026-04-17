from collections import OrderedDict


class EvalCache:
    def __init__(self, max_entries):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.max_entries = max_entries
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def size(self):
        return len(self._entries)

    def get(self, key):
        if key not in self._entries:
            self.misses += 1
            return None

        self.hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key, value):
        if key in self._entries:
            self._entries.move_to_end(key)

        self._entries[key] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
