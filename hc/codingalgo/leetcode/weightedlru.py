class ListNode:

    def __init__(self, key: str, value: int, weight: int):
        self.key = key
        self.val = value
        self.weight = weight
        self.prev = None
        self.next = None

class WeightedLRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[str, ListNode] = {}
        self.sz = 0
        self.front = ListNode("", 0, 0)
        self.back = ListNode("", 0, 0)
        self.front.next, self.back.prev = self.back, self.front

    def insert(self, pos: ListNode, node: ListNode):
        nxt = pos.next
        pos.next, node.next = node, nxt
        node.prev, nxt.prev = pos, node
        self.sz += node.weight

    def delete(self, node: ListNode):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev
        self.sz -= node.weight
    
    def get(self, key: str) -> int:
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self.delete(node)
        self.insert(self.front, node)
        return node.val

    def put(self, key: str, value: int, size: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            self.delete(node)
            self.cache.pop(key)

        while self.sz + size > self.capacity:
            evicted_node = self.back.prev
            self.delete(evicted_node)
            self.cache.pop(evicted_node.key)
        
        node = ListNode(key, value, size)
        self.cache[key] = node
        self.insert(self.front, node)


if __name__ == "__main__":
    cache = WeightedLRUCache(2)
    cache.put("1", 1, 1)
    cache.put("2", 1, 1)
    print(cache.get("1")) # 1
    cache.put("3", 1, 1)
    print(cache.get("2")) # -1, evicited
    cache.put("4", 1, 2)
    print(cache.get("1")) # -1
    print(cache.get("3")) # -1


    print("====new test====")
    cache = WeightedLRUCache(10)
    cache.put("a", 1, 3)
    cache.put("b", 2, 4)
    cache.put("c", 3, 5)

    print(cache.get("a"))
    print(cache.get("b"))
    
    cache.put("d", 4, 3)
    print(cache.cache)