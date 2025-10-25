class KVStore:

    def __init__(self):
        self.cache = {}
    
    def set(self, key: str, val: str, ts: int):
        if key not in self.cache:
            self.cache[key] = [(ts, val)]
        else:
            values = self.cache[key]
            left = 0
            right = len(values)
            while left < right:
                mid = left + (right - left) // 2
                if values[mid][0] > ts:
                    right = mid
                else:
                    left = mid + 1
            values.insert(left, (ts, val))
    
    def get(self, key: str, ts: int) -> str:
        if key not in self.cache:
            return ""
        values = self.cache[key]
        left = 0
        right = len(values)
        while left < right:
            mid = left + (right - left) // 2
            if values[mid][0] < ts:
                left = mid + 1
            else:
                right = mid
        left -= 1
        if left < 0:
            return ""
        return values[left][1]


store = KVStore()
store.set("foo", "bar", 1)
store.set("foo", "not bar", 3)
store.set("foo", "magic", 2)
print(store.cache)
print(store.get("foo", 3))
print(store.get("foo", 2))