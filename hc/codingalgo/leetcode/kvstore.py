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



class TimeMap:

    def __init__(self):
        self.cache: dict[str, list[tuple[int, str]]] = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.cache:
            self.cache[key] = [(timestamp, value)]
        else:
            kv = self.cache[key]
            left, right = 0, len(kv)
            while left < right:
                mid = left + (right - left) // 2
                if kv[mid][0] >= timestamp:
                    right = mid
                else:
                    left = mid + 1
            kv.insert(left, (timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.cache:
            return ""
        
        kv = self.cache[key]
        left, right = 0, len(kv)
        while left < right:
            mid = left + (right - left) // 2
            if kv[mid][0] > timestamp:
                right = mid
            else:
                left = mid + 1
        left -= 1
        if left < 0:
            return ""
        else:
            return kv[left][1]
        
    def save(self, path: str):
        buffer = []
        for k, v in self.cache.items():
            values = [f"{timestamp},{value}" for timestamp, value in v]
            values = ";".join(values)
            buffer.append(f"{k}:{values}")
        buffer = "\n".join(buffer)
        with open(path, 'w') as f:
            f.write(buffer)

    @classmethod
    def load(cls, path: str) -> "TimeMap":
        with open(path, 'r') as f:
            buffer = f.read()
        buffer = buffer.split("\n")
        cache = {}
        for b in buffer:
            k, values = b.split(":")
            values = values.split(";")
            values = [v.split(",") for v in values] # list of list
            values = [(int(v[0]), v[1]) for v in values]
            cache[k] = values
        
        instance = cls()
        instance.cache = cache
        return instance


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

kv = TimeMap()
kv.set("foo", "bar1", 1)
print(kv.get("foo", 2)) # 1
kv.set("foo", "bar3", 3)
kv.set("foo", "bar2", 2)
kv.set("foo", "bar4", 4)
print(kv.get("foo", 3)) # 3
print(kv.get("foo", 4)) # 4


print("===Test Save Load===")
path = "/tmp/test"
kv.save(path)

new_kv = TimeMap.load(path)
print(kv.get("foo", 2)) # 2
print(kv.get("foo", 3)) # 3
print(kv.get("foo", 4)) # 4


class TimeMap:

    def __init__(self):
        self.cache: dict[str, list[tuple[int, str]]] = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.cache:
            self.cache[key] = [(timestamp, value)]
        else:
            kv = self.cache[key]
            left, right = 0, len(kv)
            while left < right:
                mid = left + (right - left) // 2
                if kv[mid][0] >= timestamp:
                    right = mid
                else:
                    left = mid + 1
            kv.insert(left, (timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.cache:
            return ""
        
        kv = self.cache[key]
        left, right = 0, len(kv)
        while left < right:
            mid = left + (right - left) // 2
            if kv[mid][0] > timestamp:
                right = mid
            else:
                left = mid + 1
        left -= 1
        if left < 0:
            return ""
        else:
            return kv[left][1]
        
    def save(self, path: str, limit: int=1000):
        buffer = []
        for k, v in self.cache.items():
            values = [f"{timestamp},{value}" for timestamp, value in v]
            values = ";".join(values)
            buffer.append(f"{k}:{values}")
        files = []
        buffer = "\n".join(buffer)
        buffer_size = len(buffer.encode("utf-8"))
        for i in range(0, buffer_size + limit - 1, limit):
            block = buffer[i * limit: (i + 1) * limit]
            file_path = path + f'.{i}'
            with open(file_path, 'w') as f:
                f.write(block)
            files.append(file_path)
        
        metadata = path + ".metadata"
        with open(metadata, 'w') as f:
            f.write('\n'.join(files))


        

    @classmethod
    def load(cls, path: str) -> "TimeMap":
        with open(path + ".metadata", 'r') as f:
             files = f.read()
        files = files.split("\n")
        buffer = ""
        for file in files:
            with open(file, 'r') as f:
                buffer += f.read()
        buffer = buffer.split("\n")
        cache = {}
        for b in buffer:
            k, values = b.split(":")
            values = values.split(";")
            values = [v.split(",") for v in values] # list of list
            values = [(int(v[0]), v[1]) for v in values]
            cache[k] = values
        
        instance = cls()
        instance.cache = cache
        return instance


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

kv = TimeMap()
kv.set("foo", "bar1", 1)
print(kv.get("foo", 2)) # 1
kv.set("foo", "bar3", 3)
kv.set("foo", "bar2", 2)
kv.set("foo", "bar4", 4)
print(kv.get("foo", 3)) # 3
print(kv.get("foo", 4)) # 4


print("===Test Save Load===")
path = "/tmp/test"
kv.save(path)

new_kv = TimeMap.load(path)
print(kv.get("foo", 2)) # 2
print(kv.get("foo", 3)) # 3
print(kv.get("foo", 4)) # 4