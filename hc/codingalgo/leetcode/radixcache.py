class RadixNode:

    def __init__(self, prefix: list[int]):
        self.prefix = prefix
        self.childs = []
        self.is_leaf = False

    def common(self, seq: list) -> int:
        i = 0
        while i < len(self.prefix) and i < len(seq) and self.prefix[i] == seq[i]:
            i += 1
        return i
    
    def split(self, prefix: int) -> tuple["RadixNode", "RadixNode"]:
        postfix = self.prefix[prefix:]
        node = RadixNode(postfix)
        node.childs.extend(self.childs)
        node.is_leaf = self.is_leaf

        self.childs.clear()
        self.prefix = self.prefix[:prefix]
        self.childs.append(node)
        return self, node

class RadixCache:

    def __init__(self):
        self.root = RadixNode([])

    def insert(self, seq: list[int]):
        return self._insert(self.root, seq)
    
    def search(self, seq: list[int]):
        return self._search(self.root, seq)

    def _insert(self, node: RadixNode, seq: list[int]):
        if not seq:
            return
    
        for child in node.childs:
            common_prefix = child.common(seq)
            if common_prefix > 0:
                if common_prefix < len(child.prefix):
                    child, _ = child.split(common_prefix)
                postfix = seq[common_prefix:]
                if len(postfix) == 0:
                    child.is_leaf = True
                else:
                    self._insert(child, postfix)
                return
        
        child_node = RadixNode(seq)
        child_node.is_leaf = True
        node.childs.append(child_node)
        return

    def _search(self, node: RadixNode, seq: list[int]) -> bool:
        if not seq:
            return True
        
        for child in node.childs:
            common_prefix = child.common(seq)
            
            if common_prefix > 0 and common_prefix == len(child.prefix):
                postfix = seq[common_prefix:]
                if len(postfix) == 0:
                    return child.is_leaf
                
        return False

def print_tree(node, level=0):
    if level == 0:
        print("Root")
    else:
        # Create an indentation based on the depth of the tree
        indent = "    " * (level - 1)
        # Determine the label based on the is_leaf flag
        status = "Leaf" if node.is_leaf else "Internal"
        print(f"{indent}├── {node.prefix} -> {status}")
        
    for child in node.childs:
        print_tree(child, level + 1)


# Assuming your RadixNode and RadixCache classes are defined above this

if __name__ == "__main__":
    cache = RadixCache()

    print("--- Milestone 1 ---")
    cache.insert([10, 20])
    cache.insert([1, 2, 3])
    cache.insert([1, 2, 3, 4, 5, 6])
    
    print_tree(cache.root)

    print("\n--- Milestone 2 ---")
    cache.insert([1, 2, 3, 40, 50, 60, 70, 80])
    cache.insert([1, 2, 3, 40, 50, 60, 700, 800])
    
    print_tree(cache.root)
    
    print("\n--- Search Tests ---")
    print(f"Search [1, 2, 3]: {cache.search([1, 2, 3])} (Expected: False/Internal depending on if you want it to be a valid entry. In Milestone 1, it was an inserted sequence, so it should be True)")
    print(f"Search [10, 20]: {cache.search([10, 20])} (Expected: True)")
    print(f"Search [40, 50]: {cache.search([40, 50])} (Expected: False - incomplete path)")