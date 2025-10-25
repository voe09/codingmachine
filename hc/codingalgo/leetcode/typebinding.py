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