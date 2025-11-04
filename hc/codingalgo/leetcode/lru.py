from heapq import heappush, heappop

class Node:

    def __init__(self, key: str, count: int):
        self.key = key
        self.count = count
        self.prev = None
        self.next = None

    def __lt__(self, other: "Node"):
        return self.count > other.count or (self.count == other.count and self.key > other.key)

    def __repr__(self):
        return f"key: {self.key}, count: {self.count}"


class LRU:

    def __init__(self, cap: int):
        self.cap = cap
        self.cache: dict[str, Node] = {}
        # double linked list
        self.left = Node("", -1)
        self.right = Node("", -1)
        self.left.next = self.right
        self.right.prev = self.left
        # heap 
        self.heap = []

    def add_key(self, key: str):
        if key in self.cache:
            node = self.cache[key]
            node.count += 1
            self.remove_node(node)
            self.insert_node(self.left, node)
            # update heap
            # let's do lazy eval
            heappush(self.heap, (-node.count, node))
        else:
            if len(self.cache) >= self.cap:
                # pop the LRU
                node_to_delete = self.right.prev
                self.remove_node(node_to_delete)
                del self.cache[node_to_delete.key]
            
            node = Node(key, 1)
            self.cache[key] = node
            self.insert_node(self.left, node)
            heappush(self.heap, (-node.count, node))
            

    def get_count_by_key(self, key: str) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # move node to the top of the double linked list
        self.remove_node(node)
        self.insert_node(self.left, node)

        return node.count

    def get_max_count(self) -> int:
        if len(self.cache) == 0:
            return -1

        while len(self.heap) > 0:
            inv_count, node = heappop(self.heap)
            if node.key not in self.cache:
                continue
            if node.count != -inv_count:
                continue
            # find the node, move the node to the top in linked list
            self.remove_node(node)
            self.insert_node(self.left, node)
            heappush(self.heap, (-node.count, node))
            return node.count
        
        return -1

    def remove_node(self, node: Node):
        prev = node.prev
        next = node.next
        prev.next = next
        next.prev = prev
    
    def insert_node(self, prev: Node, node: Node):
        next = prev.next
        prev.next = node
        node.next = next
        next.prev = node
        node.prev = prev


lru = LRU(2)
lru.add_key("a") # a: 1
lru.add_key("b") # b:1, a:1
print(lru.get_count_by_key("a")) # 1, a:1, b:1
lru.add_key("c") # c:1, a:1
print(lru.get_count_by_key("b")) # -1
lru.add_key("a") # a: 2, c:1
print(lru.get_max_count()) # 2, #a: 2, c:1
lru.add_key("d")
print(lru.get_count_by_key("c")) # -1
print(lru.heap)


import unittest

class Node:

    def __init__(self, key: str, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRU:

    def __init__(self, cap: int):
        self.cap = cap
        self.cache: dict[str, Node] = {}
        self.first = Node("", -1)
        self.last = Node("", -1)
        self.first.next = self.last
        self.last.prev = self.first

    def add(self, key: str):
        if key in self.cache:
            node = self.cache[key]
            node.value += 1
            self._delete_node(node)
            self._insert_node(self.first, node)
        
        else:
            if len(self.cache) >= self.cap:
                # pop the least used
                node_to_delete = self.last.prev
                self._delete_node(node_to_delete)
                del self.cache[node_to_delete.key]
            
            node = Node(key, 1)
            self._insert_node(self.first, node)
            self.cache[key] = node


    def get_count(self, key: str) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._delete_node(node)
        self._insert_node(self.first, node)
        return node.value
    
    def _delete_node(self, node: Node):
        prev, next = node.prev, node.next
        prev.next = next
        next.prev = prev
    
    def _insert_node(self, prev: Node, node: Node):
        next = prev.next
        prev.next = node
        node.next = next
        next.prev = node
        node.prev = prev



class TestLRU(unittest.TestCase):

    def test_lru(self):
        cache = LRU(2)
        self.assertEqual(-1, cache.get_count("foo"))

        cache.add("foo0") # (foo: 1)
        cache.add("foo1") # (foo1: 1), (foo0: 1)
        self.assertEqual(1, cache.get_count("foo0")) # (foo0: 1), (foo1: 1)
        cache.add("foo2") # (foo2: 1), (foo0: 1)
        self.assertEqual(-1, cache.get_count("foo1"))
        cache.add("foo0")
        self.assertEqual(2, cache.get_count("foo0"))


unittest.main(verbosity=2)

    