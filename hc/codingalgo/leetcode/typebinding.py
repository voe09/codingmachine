from typing import Union

class Node:

    def __init__(self, val: Union[str, list["Node"]]):
        self.primitive_types = set(["str", "int", "float"])
        
        if isinstance(val, str):
            self.base = val
            self.children = []
        else:
            self.base = None
            self.children = val
    
    def is_primitive_type(self):
        return self.base is not None and self.base in self.primitive_types

    def is_generic_type(self):
        return self.base is not None and self.base not in self.primitive_types

    def is_composity_type(self):
        return self.base is None and len(self.children) > 0
    
    def __str__(self) -> str:
        if self.base is not None:
            return self.base
        
        types = []
        for child in self.children:
            types.append(str(child))
        
        return f"[{','.join(types)}]"


class Function:

    def __init__(self, args: list[Node], output: Node):
        self.args = args
        self.output = output

    def __str__(self):
        input_types = [str(arg) for arg in self.args]
        return f"({','.join(input_types)}) -> {str(self.output)}"


def helper(node1: Node, node2: Node, binding_map: dict) -> bool:
    if node1.is_composity_type() != node2.is_composity_type():
        return False

    if not node1.is_composity_type():
        # if node2 is not generic type, it must match node1's base type
        if not node2.is_generic_type():
            if node2.base != node1.base:
                return False
            else:
                return True
        # if node2 is generic type, no conflict on binding map
        else:
            if node2.base not in binding_map:
                binding_map[node2.base] = node1.base
                return True
            else:
                if binding_map[node2.base] != node1.base:
                    return False
                else:
                    return True
    else:
        for child1, child2 in zip(node1.children, node2.children):
            if not helper(child1, child2, binding_map):
                return False
    
    return True


def resolve_generic(node: Node, binding_map: dict) -> Node:
    if node.is_composity_type():
        for child in node.children:
            resolve_generic(child, binding_map)

    else:
        if node.is_generic_type():
            node.base = binding_map[node.base]
    return node    

def get_return_type(args: list[Node], func: Function) -> Node:
    if any([arg.is_generic_type() for arg in args]):
        raise ValueError("generic type is not allowed in get_return_type")

    if len(args) != len(func.args):
        raise ValueError("arg len must be equal")

    binding_map = {}
    for arg, func_arg in zip(args, func.args):
        if not helper(arg, func_arg, binding_map):
            raise ValueError("invalid binding")
        
    return resolve_generic(func.output, binding_map)


# (float, [str, int])
# (S, [str, T]) -> (T, S)
# output (int, float)

arg1 = Node("float")
arg2 = Node([Node("str"), Node("int")])

type_arg1 = Node("S")
type_arg2 = Node([Node("str"), Node("T")])
type_output = Node([Node("T"), Node("S")])

func = Function([type_arg1, type_arg2], type_output)

print(str(func))

output = get_return_type([arg1, arg2], func)
print(output)



import unittest

from typing import Optional
from unittest.loader import VALID_MODULE_NAME

class Node:

    def __init__(
        self, 
        value: Optional[str] = None, 
        children: Optional[list["Node"]] = None,
    ):
        self.value = value
        self.children = children

    def is_nested(self):
        return self.children is not None and len(self.children) > 0
    
    def is_primitive(self):
        return self.value is not None and self.value in ["char", "int", "float"]

    def to_str(self):
        if self.value is not None:
            return self.value
        
        else:
            return f"[{','.join([c.to_str() for c in self.children])}]"

    def resolve(self, mapping: dict[str, str]) -> "Node":
        new_node = Node()
        if self.is_nested():
            new_node.children = []
            for child in self.children:
                new_node.children.append(child.resolve(mapping))
        else:
            if self.is_primitive():
                new_node.value = self.value
            else:
                generic_type = self.value
                if generic_type not in mapping:
                    raise ValueError(f"{self.to_str()} is not resolveable, func signature is incorrect")
                else:
                    new_node.value = mapping[generic_type]
        
        return new_node

class Function:

    def __init__(self, args: list[Node], output: Node):
        self.args = args
        self.output = output

    def to_str(self):
        return f"[{','.join([arg.to_str() for arg in self.args])}] -> {self.output.to_str()}"

    def resolve(self, params: list[Node]) -> Node:
        if len(self.args) != len(params):
            raise ValueError("input params length does not match function signature")

        mapping: dict[str, str] = {}

        for arg, param in zip(self.args, params):
            helper(arg, param, mapping)
        
        return self.output.resolve(mapping)

def helper(a: Node, b: Node, mapping: dict[str, str]):
    if a.is_nested() != b.is_nested():
        raise ValueError(f"type mismatch for {a.to_str()} and {b.to_str()}")

    if a.is_nested():
        if len(a.children) != len(b.children):
            raise ValueError(f"type mismatch for {a.to_str()} and {b.to_str()}")
        for child1, child2 in zip(a.children, b.children):
            helper(child1, child2, mapping)
        
    else:
        if not a.is_primitive():
            generic_type = a.value
            primitive_type = b.value
            if generic_type in mapping:
                if mapping[generic_type] != primitive_type:
                    raise ValueError(f"generic type mismatch, {generic_type} maps to both {primitive_type} and {mapping[generic_type]}")
            else:
                mapping[generic_type] = primitive_type
        else:
            if a.value != b.value:
                raise ValueError(f"type mismatch for {a.to_str()} and {b.to_str()}")


class TestToStr(unittest.TestCase):

    def setUp(self) -> None:
        self.node1 = Node("T1")
        self.node2 = Node("char")
        self.node3 = Node(children=[self.node1, self.node2])
        self.node4 = Node(children=[self.node1, self.node3])

    def test_node_to_str(self):
        self.assertEqual("T1", self.node1.to_str())
        self.assertEqual("[T1,char]", self.node3.to_str())
        self.assertEqual("[T1,[T1,char]]", self.node4.to_str())

    def test_function_to_str(self):
        function1 = Function([self.node2, self.node4], self.node1)
        self.assertEqual("[char,[T1,[T1,char]]] -> T1", function1.to_str())

        function2 = Function([self.node1, self.node3], self.node4)
        self.assertEqual("[T1,[T1,char]] -> [T1,[T1,char]]", function2.to_str())


class TestFunctionResolve(unittest.TestCase):

    def setUp(self) -> None:
        self.node1 = Node("T1")
        self.node2 = Node("char")
        self.node3 = Node(children=[self.node1, self.node2]) # T1,char
        self.node4 = Node(children=[self.node1, self.node3])

        self.node5 = Node("int") # map to node 1
        self.node6 = Node(children=[self.node5, self.node2]) # map to node 3
        self.node7 = Node(children=[self.node5, self.node6]) # map to node 4

        self.node8 = Node(children=[self.node5, self.node5, self.node2]) # error mapping for node3 len mismatch

        self.node9 = Node(children = [self.node2, self.node2]) # T1 match char

    def test_function_resolve_the_type(self):
        # function has 3 args: one generic type, one primitive type and one nested type
        function = Function([self.node1, self.node2, self.node4], self.node3)
        output_type = function.resolve([self.node5, self.node2, self.node7])
        self.assertEqual("[int,char]", output_type.to_str())

    def test_param_len_mismatch(self):
        function = Function([self.node1, self.node2], self.node3)
        with self.assertRaises(ValueError) as ve:
            function.resolve([self.node5])
        print(ve.exception)
    
    def test_param_children_len_mismatch(self):
        function = Function([self.node1, self.node3], self.node3)
        with self.assertRaises(ValueError) as ve:
            function.resolve([self.node1, self.node8])
        print(ve.exception)

    def test_primitive_type_mismatch(self):
        function = Function([self.node2, self.node3], self.node3)
        with self.assertRaises(ValueError) as ve:
            function.resolve([self.node5, self.node6])
        print(ve.exception)

    def test_generic_type_multi_match(self):
        function = Function([self.node1, self.node3], self.node3)
        with self.assertRaises(ValueError) as ve:
            function.resolve([self.node5, self.node9])
        print(ve.exception)

unittest.main(verbosity=2)