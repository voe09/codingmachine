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












from typing import Optional


class Cell:

    def __init__(self, key: str, value: Optional[int] = None):
        self.key = key
        self.value = value
        self.resolved_value = None
        self.parents = set()
        self.children = set()

    def add_parent(self, node: "Cell"):
        self.parents.add(node)
        node.children.add(self)

    def remove_parent(self, node: "Cell"):
        self.parents.remove(node)
        node.children.remove(self)

    def resolve(self):
        # resolve value from parents
        if self.value is None:
            value = None if any(p.resolved_value is None for p in self.parents) else sum(p.resolved_value for p in self.parents)
            self.resolved_value = value
        else:
            self.resolved_value = self.value

        for child in self.children:
            child.resolve()

    def validate(self, visited: set):
        for p in self.parents:
            if p in visited:
                raise ValueError("Circle Detected")
            visited.add(p)
            p.validate(visited)
            visited.remove(p)

    def __repr__(self) -> str:
        return f"{self.key}"

class SpreadSheet:

    def __init__(self):
        self.cells = {}
    
    def set_cell(
        self, 
        key: str, 
        value: Optional[int] = None, 
        child1: Optional[str] = None, 
        child2: Optional[str] = None,
    ):
        if child1 is not None and child1 not in self.cells:
            self.cells[child1] = Cell(child1)
        if child2 is not None and child2 not in self.cells:
            self.cells[child2] = Cell(child2)

        if key not in self.cells:
            cell = Cell(key, value)
            self.cells[key] = cell
            if child1 is not None:
                cell.add_parent(self.cells[child1])
                cell.add_parent(self.cells[child2])
        else:
            origin_cell = self.cells[key]

            for p in list(origin_cell.parents):
                    origin_cell.remove_parent(p)

            origin_cell.value = None

            if value is not None:
                origin_cell.value = value
            else:
                origin_cell.add_parent(self.cells[child1])
                origin_cell.add_parent(self.cells[child2])
        
        visited = set([self.cells[key]])
        self.cells[key].validate(visited)

        self.cells[key].resolve()

    def get_cell(self, key: str):
        if key not in self.cells:
            return -1
        return self.cells[key].resolved_value



sheet = SpreadSheet()
sheet.set_cell("A", 1)
sheet.set_cell("B", 2)
sheet.set_cell("C", child1="A", child2="B")
print(sheet.get_cell("A"))
print(sheet.get_cell("B"))
print(sheet.get_cell("C"))
sheet.set_cell("D", child1="A", child2="C")
print(sheet.get_cell("D"))
sheet.set_cell("C", 10)
print(sheet.get_cell("A"))
print(sheet.get_cell("B"))
print(sheet.get_cell("C"))
print(sheet.get_cell("D")) # D is not correctly updated

print(sheet.cells)
for cell in sheet.cells:
    print(f"node: {sheet.cells[cell]}, its parents: {sheet.cells[cell].parents}")
sheet.set_cell("A", child1="C", child2="B")





import unittest

from typing import Optional


class Node:

    def __init__(self, key: str, value: Optional[int] = None):
        self.key = key
        self.value = value
        self.children = set()
        self.parent = set()
        self.resolved_value = None
    
    def add(self, child: "Node"):
        self.children.add(child)
        child.parent.add(self)

    def remove(self, child: "Node"):
        self.children.remove(child)
        child.parent.remove(self)
    
    def invalidate(self):
        if self.resolved_value is not None:
            self.resolved_value = None
            for p in self.parent:
                p.invalidate()

    def validate(self, visited: set["Node"]) -> bool:
        if not self.children:
            return True
        
        for child in self.children:
            if child in visited:
                return False
            visited.add(child)
            child.validate(visited)
            visited.remove(child)
        
        return True

    def getValue(self):
        if self.value:
            return self.value
        elif self.resolved_value:
            return self.resolved_value
        else:
            resolved_value = sum(child.getValue() for child in self.children)
            self.resolved_value = resolved_value
            return resolved_value

    
class Spreadsheet:

    def __init__(self):
        self.cells = {}

    def setCell(
        self, 
        key: str, 
        value: Optional[int] = None, 
        child1: Optional[str] = None, 
        child2: Optional[str] = None,
    ):
        if child1 is not None:
            self.cells[child1] = Node(child1)
        if child2 is not None:
            self.cells[child2] = Node(child2)
        if key not in self.cells:
            cell = Node(key, value)
            if child1:
                cell.add(self.cells[child1])
                cell.add(self.cells[child2])
            self.cells[key] = cell
        else:
            cell = self.cells[key]
            # remove the original child
            for child in list(cell.children):
                cell.remove(child)
            cell.invalidate() # invalidate cache
            cell.value = value
            if child1:
                cell.add(self.cells[child1])
                cell.add(self.cells[child2])
            
        visited = set([self.cells[key]])
        if not self.cells[key].validate(visited):
            raise IndexError("circle detected")

    def getCell(self, key: str) -> Optional[int]:
        if key not in self.cells:
            return None
        else:
            return self.cells[key].getValue()



class TestSpreadsheet(unittest.TestCase):

    def setUp(self) -> None:
        self.sheet = Spreadsheet()
    
    def set_cell_and_get_cell(self):
        self.sheet.setCell("A1", 100)
        self.sheet.setCell("B1", 200)
        self.sheet.setCell("C1", None, "A1", "B1")

        self.assertEqual(100, self.sheet.getCell("A1"))
        self.assertEqual(200, self.sheet.getCell("B1"))
        self.assertEqual(300, self.sheet.getCell("C1"))

        self.assertIsNone(self.sheet.getCell("Not a key"))

    
    def set_cell_and_reset_cell(self):
        self.sheet.setCell("C1", None, "A1", "B1")
        self.sheet.setCell("A1", 100)
        self.sheet.setCell("B1", 200)
        self.assertEqual(300, self.sheet.getCell("C1"))
        self.sheet.setCell("D1", None, "C1", "A1")
        self.assertEqual(400, self.sheet.getCell("D1"))

        self.sheet.setCell("C1", None, "A2", "B1")
        self.sheet.setCell("A2", 500)
        self.assertEqual(700, self.sheet.getCell("C1"))
        self.assertEqual(800, self.sheet.getCell("D1"))

    def circle(self):
        self.sheet.setCell("A1", None, "B1", "C1")
        self.sheet.setCell("B1", 100)
        with self.assertRaises(IndexError):
            self.sheet.setCell("C1", None, "A1", "B1")

unittest.main(verbosity=2)