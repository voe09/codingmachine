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



class KVCache:

    def __init__(self):
        self.cache = {}

    def set(self, key: str, value: str, timestamp: int):
        if key not in self.cache:
            self.cache[key] = [(timestamp, value)]
        else:
            data = self.cache[key]
            left, right = 0, len(data)
            while left < right:
                mid = left + (right - left) // 2
                if data[mid][0] > timestamp:
                    right = mid
                else:
                    left = mid + 1
            
            # pos = bisect_right(data, timestamp, key = lambda x: x[0])
            data.insert(left, (timestamp, value))

    def get(self, key: str, timestamp: int):
        if key not in self.cache:
            return ""
        
        data = self.cache[key]
        left, right = 0, len(data)
        while left < right:
            mid = left + (right - left) // 2
            if data[mid][0] > timestamp:
                right = mid
            else:
                left = mid + 1
        pos = left
        # pos = bisect_right(data, timestamp, key=lambda x: x[0])
        if pos == 0:
            return ""
        else:
            return data[pos-1][1]

    # def save(self) -> str:
    #     buffer = []
    #     for k, v in self.cache.items():
    #         values = [f"{ts},{value}" for ts, value in v]
    #         buffer.append(f"{k}:{';'.join(values)}")
    #     return '\n'.join(buffer)

    # @classmethod
    # def load(cls, data: str):
    #     ins = cls()
    #     def parse():
    #         cache = {}
    #         buffer = data.split('\n')
    #         for line in buffer:
    #             key, values = line.split(':')
    #             values = values.split(';')
    #             values = [v.split(',') for v in values]
    #             values = [(int(ts), v) for ts, v in values]
    #             cache[key] = values
    #         return cache

    #     ins.cache = parse()
    #     return ins

    def save(self, limit: int) -> list[str]:
        res = []
        buffer = []
        buffer_size = 0
        for k, v in self.cache.items():
            values = [f"{ts},{v}" for ts, v in v]
            line = f"{k}:{';'.join(values)}"
            line_size = len(line.encode("utf-8"))
            if buffer_size + line_size + len(buffer) - 1 > limit:
                res.append('\n'.join(buffer))
                buffer = [line]
                buffer_size = line_size
            else:
                buffer.append(line)
                buffer_size += line_size
        if len(buffer):
            res.append('\n'.join(buffer))
        return res

    @classmethod
    def load(cls, data: list[str]):
        ins = cls()

        def parse_batch(batch: str):
            cache = {}
            buffer = batch.split('\n')
            for line in buffer:
                key, values = line.split(':')
                values = values.split(';')
                values = [v.split(',') for v in values]
                values = [(int(ts), v) for ts, v in values]
                cache[key] = values
            return cache

        cache = {}
        for batch in data:
            cache.update(parse_batch(batch))
        
        ins.cache = cache
        return ins


cache = KVCache()
cache.set("foo", "bar", 1)
print(cache.get("foo", 1))
cache.set("foo", "bar2", 4)
print(cache.get("foo", 3))
print(cache.get("foo", 4))
print(cache.get("foo", 5))

cache.set("key", "value", 10)

state = cache.save(limit=20)
print(state)

new_cache = KVCache.load(state)
print(new_cache.cache)


import unittest
import threading

from typing import Optional
from abc import ABC, abstractmethod

class KeyValueStore(ABC):
    @abstractmethod
    def put(self, key: str, value: str) -> int:
        """Inserts a new value for the given key and returns the version number."""
        pass

    @abstractmethod
    def get(self, key: str, version: Optional[int] = -1) -> Optional[str]:
        """Retrieves the value associated with the key at the given version.
        If version is -1, return the latest value."""
        pass

class VersionKVStore(KeyValueStore):

    def __init__(self):
        self.store: dict[str, dict[int, str]] = {}

    def put(self, key: str, value: str) -> int:
        if key not in self.store:
            self.store[key] = {0: value}
            return 0
        else:
            latest_version = len(self.store[key])
            self.store[key][latest_version] = value
            return latest_version

    def get(self, key:str, version: Optional[int] = -1) -> Optional[str]:
        if key not in self.store:
            return None
        
        else:
            versioned_values = self.store[key]
            if version is not None and version != -1:
                if version not in versioned_values:
                    return None
                else:
                    return versioned_values[version]
            else:
                return versioned_values[len(versioned_values) - 1]
            

class ThreadsafeVersionKVStore(KeyValueStore):

    def __init__(self):
        self.store: dict[str, dict[int, str]] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()

    def put(self, key: str, value: str) -> int:
        with self.global_lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            key_lock = self.locks[key]

        with key_lock:
            if key not in self.store:
                self.store[key] = {0: value}
                return 0
            else:
                latest_version = len(self.store[key])
                self.store[key][latest_version] = value
                return latest_version

    def get(self, key:str, version: Optional[int] = -1) -> Optional[str]:
        with self.global_lock:
            if key not in self.locks:
                return None
            key_lock = self.locks[key]
        
        with key_lock:
            versioned_values = self.store[key]
            if version is not None and version != -1:
                if version not in versioned_values:
                    return None
                else:
                    return versioned_values[version]
            else:
                return versioned_values[len(versioned_values) - 1]

class TestVersionedKVStore(unittest.TestCase):

    def setUp(self):
        self.cache = VersionKVStore()

    def test_put_and_get(self):
        self.cache.put("foo", "bar0")
        self.cache.put("foo", "bar1")
        self.cache.put("foo", "bar2")
        self.assertEqual(1, len(self.cache.store))
        self.assertEqual(3, len(self.cache.store["foo"]))
        self.cache.put("notfoo", "bar")
        self.assertEqual(2, len(self.cache.store))

        self.assertIsNone(self.cache.get("not a key"))
        self.assertIsNone(self.cache.get("foo", 3))
        self.assertEqual("bar1", self.cache.get("foo", 1))
        self.assertEqual("bar2", self.cache.get("foo", -1))


class TestThreadsafeVersionedKVStore(unittest.TestCase):
    
    def setUp(self) -> None:
        self.cache = ThreadsafeVersionKVStore()

    def test_put_and_get_threadsafe(self):
        jobs = []
        for i in range(100):
            job = threading.Thread(target=self.cache.put, args=("foo", f"bar{i}",))
            jobs.append(job)

        def get(key: str, version: int):
            v = self.cache.get(key, version)
            print(v)

        for i in range(100):
            job = threading.Thread(target=get, args=("foo", i))
            jobs.append(job)

        for job in jobs:
            job.start()

        done = [job.join() for job in jobs]

        print(done)



unittest.main(verbosity=2)

    