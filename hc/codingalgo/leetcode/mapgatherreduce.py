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