class TreeNode:

    def __init__(self, value: int):
        self.value = value
        self.children = []

    def add_child(self, node: "TreeNode") -> bool:
        if node.value >= self.value:
            return False

        self.children.append(node)

    def check(self) -> tuple[int, bool]:
        if len(self.children) == 0:
            return self.value, True

        mx = float("-inf")
        for child in self.children:
             val, valid = child.check()
             if not valid:
                return self.value, False
             mx = max(mx, val)
            
        if self.value <= mx:
            return mx, False
        
        return self.value, True


node = TreeNode(5)
print(node.add_child(TreeNode(6)))
node.add_child(TreeNode(4))
node.children[0].add_child(TreeNode(3))
print(node.check())

