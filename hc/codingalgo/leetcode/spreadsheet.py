# spreadsheet setCell, getCell
from typing import Optional

class Cell:

    def __init__(
        self,
        node_id: str,
        value: Optional[int] = None, 
        parents: Optional[list["Cell"]] = None,
    ):
        self.node_id = node_id
        self.value = value
        self.parents = parents
        self.children = {}

        if self.parents:
            # Build reverse link
            for cell in self.parents:
                cell.children[self.node_id] = self
    
    def is_composite_type(self):
        return self.parents is not None

    def set_cell(
        self, 
        value: Optional[int] = None, 
        parents: Optional[list["Cell"]] = None,
    ):
        if self.is_composite_type():
            # remove the node in the graph
            for parent in self.parents:
                del parent.children[self.node_id]
            if value is not None:
                self.value = value
            else:
                self.parents = parents
                for parent in self.parents:
                    parent.children[self.node_id] = self
        else:
            if value is not None:
                self.value = value
            else:
                self.value = None
                self.parents = parents
                for parent in self.parents:
                    parent.children[self.node_id] = self


    def get_cell(self) -> int:
        if self.value:
            return self.value
        
        val = 0
        for parent in self.parents:
            visited = set()
            visited.add(parent.node_id)
            val += self.visit(parent, visited)
        
        return val
    
    def visit(self, node: "Cell", visited: set[str]) -> int:
        if node.value:
            return node.value
        
        val = 0
        for parent in node.parents:
            if parent in visited:
                raise ValueError("circle")
            visited.add(parent.node_id)
            val += self.visit(parent, visited)
        return val


class Spreadsheet:

    def __init__(self):
        self.nodes = {}

    def set_cell(self, node_id: str, expression: str):
        if expression.startswith("="):
            node_ids = expression.lstrip("=").split("+")
            cells = []
            for id in node_ids:
                if id not in self.nodes:
                    raise ValueError("invalid arg")
                cells.append(self.nodes[id])
            if node_id not in self.nodes:
                self.nodes[node_id] = Cell(node_id=node_id, parents=cells)
            else:
                self.nodes[node_id].set_cell(parents=cells)
        else:
            val = int(expression)
            if node_id not in self.nodes:
                self.nodes[node_id] = Cell(node_id, value=val)
            else:
                self.nodes[node_id].set_cell(value=val)

    def get_cell(self, node_id: str):
        if node_id not in self.nodes:
            raise ValueError("cell not exist")
        
        return self.nodes[node_id].get_cell()


spreadsheet = Spreadsheet()
spreadsheet.set_cell("A1", "1")
spreadsheet.set_cell("A2", "2")
spreadsheet.set_cell("A3", "=A1+A2") # 3
spreadsheet.set_cell("A4", "=A3+A1+A2") # 6
print(spreadsheet.get_cell("A1"))
print(spreadsheet.get_cell("A2"))
print(spreadsheet.get_cell("A3"))
print(spreadsheet.get_cell("A4"))
spreadsheet.set_cell("A4", "=A3+A3+A1") # 7
print(spreadsheet.get_cell("A4"))




# spreadsheet setCell, getCell
from typing import Optional

class Cell:

    def __init__(
        self,
        node_id: str,
        value: Optional[int] = None, 
        parents: Optional[list["Cell"]] = None,
    ):
        self.node_id = node_id
        self.value = value
        self.parents = parents
        self.children = {}

        if self.parents:
            # Build reverse link
            for cell in self.parents:
                cell.children[self.node_id] = self
        
        self.resolved_value = None
        self.resolve()

    def resolve(self):
        if self.value is not None:
            self.resolved_value = self.value
            return
    
        val = 0
        for cell in self.parents:
            if cell.resolved_value is None:
                self.resolved_value = None
                return
            else:
                val += cell.resolved_value
        self.resolved_value = val

    def is_composite_type(self):
        return self.parents is not None

    def set_cell(
        self, 
        value: Optional[int] = None, 
        parents: Optional[list["Cell"]] = None,
    ):
        # unlink original parents
        if self.parents:
            for parent in self.parents:
                del parent.children[self.node_id]

        self.value = value
        self.parents = parents
        # relink parents
        if self.parents:
            for parent in self.parents:
                parent.children[self.node_id] = self
    
        self.resolve()

        # update children's resolved value
        visited = set()
        visited.add(self.node_id)
        for child in self.children.values():
            if child.node_id in visited:
                raise ValueError("circle")
            visited.add(child.node_id)
            child.resolve()
            child.propogate(visited)
            visited.remove(child.node_id)

    def propogate(self, visited: set[str]):
        for child in self.children.values():
            if child.node_id in visited:
                raise ValueError("circle")
            visited.add(child.node_id)
            child.resolve()
            child.propogate(visited)
            visited.remove(child.node_id)

    def get_cell(self) -> int:
        return self.resolved_value

class Spreadsheet:

    def __init__(self):
        self.nodes = {}

    def set_cell(self, node_id: str, expression: str):
        if expression.startswith("="):
            node_ids = expression.lstrip("=").split("+")
            cells = []
            for id in node_ids:
                if id not in self.nodes:
                    raise ValueError("invalid arg")
                cells.append(self.nodes[id])
            if node_id not in self.nodes:
                self.nodes[node_id] = Cell(node_id=node_id, parents=cells)
            else:
                self.nodes[node_id].set_cell(parents=cells)
        else:
            val = int(expression)
            if node_id not in self.nodes:
                self.nodes[node_id] = Cell(node_id, value=val)
            else:
                self.nodes[node_id].set_cell(value=val)

    def get_cell(self, node_id: str):
        if node_id not in self.nodes:
            raise ValueError("cell not exist")
        
        return self.nodes[node_id].get_cell()


spreadsheet = Spreadsheet()
spreadsheet.set_cell("A1", "1")
spreadsheet.set_cell("A2", "2")
spreadsheet.set_cell("A3", "=A1+A2") # 3
spreadsheet.set_cell("A4", "=A3+A1+A2") # 6
print(spreadsheet.get_cell("A1"))
print(spreadsheet.get_cell("A2"))
print(spreadsheet.get_cell("A3"))
print(spreadsheet.get_cell("A4"))
spreadsheet.set_cell("A4", "=A3+A3+A1") # 7
print(spreadsheet.get_cell("A4"))
spreadsheet.set_cell("A5", "=A4+A3+A3+A1") # 7 + 3 + 3 + 1 = 14
print(spreadsheet.get_cell("A5"))
spreadsheet.set_cell("A1", "100")
print(spreadsheet.get_cell("A1"))
print(spreadsheet.get_cell("A2"))
print(spreadsheet.get_cell("A3"))
print(spreadsheet.get_cell("A4"))
print(spreadsheet.get_cell("A5"))
spreadsheet.set_cell("A1", "=A5+A1")