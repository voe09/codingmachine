from typing import Optional

NODE_REGISTRY: dict[str, "Node"] = {}

class Node:

    def __init__(
        self, 
        node_id: str):
        self.node_id = node_id
        self.parent = None
        self.children = []

        self.buffer = []

    def add_parent(self, parent: str):
        self.parent = parent

    def add_child(self, child: str):
        if self.children is None:
            self.children = []

        self.children.append(child)
    
    def sendAsyncMessage(self, node_id: str, msg: str):
        NODE_REGISTRY[node_id].receiveMessage(self.node_id, msg)

    def receiveMessage(self, node_id: str, msg: str):
        msg_type = "COUNT" if (self.parent is not None and node_id == self.parent) or (self.parent is None and node_id not in self.children) else "COUNT_RESPONSE"

        if msg_type == "COUNT": # map
            # leaf node
            if self.children is None and self.parent is None:
                print("TOTAL COUNT: 1")
                return
            elif self.children is None:
                self.sendAsyncMessage(self.parent, "1")
            else:
                for child in self.children:
                    self.sendAsyncMessage(child, "-1")
        else: # COUNT RESPONSE - gather
            count = int(msg)
            self.buffer.append((node_id, count))

            if len(self.buffer) == len(self.children):
                total_count = sum([b[1] for b in self.buffer]) + 1
                if self.parent is None:
                    print(f"TOTAL COUNT: {total_count}")
                else:
                    self.sendAsyncMessage(self.parent, str(total_count))
            

#    1
#   2  3
#  4  5 #  6


nodes = [Node(str(i)) for i in range(1, 7)]
for node in nodes:
    NODE_REGISTRY[node.node_id] = node

nodes[0].add_child(nodes[1].node_id)
nodes[1].add_parent(nodes[0].node_id)
nodes[0].add_child(nodes[2].node_id)
nodes[2].add_parent(nodes[0].node_id)

nodes[1].add_child(nodes[3].node_id)
nodes[1].add_child(nodes[4].node_id)
nodes[3].add_parent(nodes[1].node_id)
nodes[4].add_parent(nodes[1].node_id)

nodes[2].add_child(nodes[5].node_id)
nodes[5].add_parent(nodes[2].node_id)


nodes[0].sendAsyncMessage("1", "-1")

NODE_REGISTRY = {}
node = Node(0)
NODE_REGISTRY["0"] = node
node.sendAsyncMessage("0", "-1")





NODE_REGISTRY: dict[str, "Node"] = {}

class Node:

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.parent = None
        self.children = []

        self.buffer = {}

    def add_parent(self, parent: str):
        self.parent = parent

    def add_child(self, child: str):
        self.children.append(child)

    def sendAsyncMessage(self, to_node_id: str, msg: str):
        NODE_REGISTRY[to_node_id].receiveMessage(self.node_id, msg)

    def receiveMessage(self, from_node_id: str, msg: str):
        is_root = True if self.parent is None else False
        is_leaf = True if len(self.children) == 0 else False

        if is_root and is_leaf:
            print(f"{self.node_id}")
            return

        msg_type = "MAP" if (not is_root and from_node_id == self.parent) or (is_root and from_node_id not in self.children) else "REDUCE"

        if msg_type == "MAP":
            if is_leaf:
                self.sendAsyncMessage(self.parent, f"{self.node_id}")
            else:
                for child in self.children:
                    self.sendAsyncMessage(child, "")

        else:
            self.buffer[from_node_id] = f"({msg})"

            if len(self.buffer) == len(self.children):
                graph = ",".join([self.node_id] + list(self.buffer.values()))
                if is_root:
                    # print topological graph
                    print(graph) 
                else:
                    self.sendAsyncMessage(self.parent, graph)


nodes = [Node(str(i)) for i in range(1, 7)]
for node in nodes:
    NODE_REGISTRY[node.node_id] = node

nodes[0].add_child(nodes[1].node_id)
nodes[1].add_parent(nodes[0].node_id)
nodes[0].add_child(nodes[2].node_id)
nodes[2].add_parent(nodes[0].node_id)

nodes[1].add_child(nodes[3].node_id)
nodes[1].add_child(nodes[4].node_id)
nodes[3].add_parent(nodes[1].node_id)
nodes[4].add_parent(nodes[1].node_id)

nodes[2].add_child(nodes[5].node_id)
nodes[5].add_parent(nodes[2].node_id)


nodes[0].sendAsyncMessage("1", "")

NODE_REGISTRY = {}
node = Node(0)
NODE_REGISTRY["0"] = node
node.sendAsyncMessage("0", "")



from typing import Optional

class TreeNode:

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.parent = None
        self.children = None
        self.buffer = []

    def add_parent(self, node: "TreeNode"):
        self.parent = node
    
    def add_children(self, children: list["TreeNode"]):
        self.children = children
        for child in self.children:
            child.add_parent(self)
        
    def sendAsyncMessage(self, node: "TreeNode", msg: str):
        node.receiveMessage(self, msg)

    def receiveMessage(self, from_node: "TreeNode", msg: str):
        if msg == "count":
            if self.children is None:
                if self.parent:
                    self.sendAsyncMessage(self.parent, "countResponse:1")
                else:
                    print("TOTAL COUNT:1")
            else:
                for child in self.children:
                    self.sendAsyncMessage(child, "count")

        elif msg.startswith("countResponse"):
            _, count = msg.split(":")
            count = int(count)
            self.buffer.append(count)

            if len(self.buffer) == len(self.children):
                count = sum(self.buffer) + 1
                if self.parent:
                    self.sendAsyncMessage(self.parent, f"countResponse:{count}")
                    self.buffer.clear()
                else:
                    print(f"TOTAL COUNT:{count}")
                    self.buffer.clear()

        elif msg == "topology":
            if self.children is None:
                if self.parent:
                    self.sendAsyncMessage(self.parent, f"topologyResponse:{self.node_id}")
                else:
                    print(f"TOPOLOGY: {self.node_id}")
            else:
                for child in self.children:
                    self.sendAsyncMessage(child, "topology")
        elif msg.startswith("topologyResponse"):
            _, topo = msg.split(":")
            self.buffer.append(topo)

            if len(self.buffer) == len(self.children):
                graph = f"({self.node_id}, ({','.join(self.buffer)}))"
                if self.parent:
                    self.sendAsyncMessage(self.parent, f"topologyResponse:{graph}")
                    self.buffer.clear()
                else:
                    print(f"TOPOLOGY: {graph}")
                    self.buffer.clear()




root = TreeNode("1")
root.sendAsyncMessage(root, "count")
root.sendAsyncMessage(root, "topology")

node1 = TreeNode("2")
node1.add_children([TreeNode("4"), TreeNode("5")])
root.add_children([node1, TreeNode("3")])

root.sendAsyncMessage(root, "count")
root.sendAsyncMessage(root, "topology")