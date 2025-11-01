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