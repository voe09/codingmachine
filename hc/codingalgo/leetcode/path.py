def cd(curr_dir: str, new_dir: str, home: str = "/home/usr") -> str:
    if new_dir.startswith("~"):
        new_dir = new_dir.replace("~", home, 1)
    if new_dir.startswith("/"):
        return simplify_path(new_dir)
    return simplify_path(curr_dir + "/" + new_dir)

def simplify_path(path: str) -> str:
    pathes = path.split("/")

    res = []
    for p in pathes:
        if p == "":
            continue
        elif p == ".":
            continue
        elif p == "..":
            if len(res) > 0:
                res.pop()
            else:
                return "NULL"
        else:
            res.append(p)
    return "/" + "/".join(res)

print(cd("/foo/bar", "baz"))
print(cd("/foo/bar", "../baz"))
print(cd("/foo/bar", "./baz"))
print(cd("/foo/bar", "../../../baz"))
print(cd("/foo/bar", "~/foo/bar"))



from typing import Optional

def simplify_path(path: str) -> str:
    tokens = path.split("/")
    stack = []
    for token in tokens:
        if token == "" or token == ".":
            continue
        elif token == "..":
            if stack:
                stack.pop()
            else:
                raise ValueError("Invalid Path")
        stack.append(token)
    return "/" + "/".join(stack)


class TreeNode:

    def __init__(self, token: str):
        self.token = token
        self.children = {}
        self.path = None

    def build(self, origin: str, new: str):
        tokens = [t for t in origin.split("/") if t]
        curr = self
        for i in range(len(tokens)):
            token = tokens[i]
            if token not in curr.children:
                curr.children[token] = TreeNode(token)
            curr = curr.children[token]
            if i == len(tokens) - 1:
                curr.path = new
    
    def visit(self, path: str) -> Optional[tuple[str, str]]:
        tokens = [t for t in path.split("/") if t]
        curr = self
        longest = None
        visited = []

        for token in tokens:
            if token in curr.children:
                visited.append(token)
                curr = curr.children[token]
                if curr.path:
                    longest = ("/" + "/".join(visited), curr.path)
            else:
                break

        return longest

def cd(curr: str, new: str, links: dict[str, str]) -> str:
    if new.startswith("/"):
        path = simplify_path(new)
    else:
        path = simplify_path(curr + "/" + new)
    
    root = TreeNode("")
    for k, v in links.items():
        root.build(k, v)
    
    visited = set()
    while True:
        if path in visited:
            raise ValueError("Circle detected")

        visited.add(path)

        match = root.visit(path)
        if match is None:
            break        
        
        origin, replace = match
        path = replace + path[len(origin):]

        path = simplify_path(path)

    return path


print(cd("/foo/bar", "baz", {"/foo/bar": "/abc", "/foo/bar/baz": "/xyz"}))




import unittest

def simplify(path: str) -> str:
    stack = []
    path = path.split('/')
    for token in path:
        if token == ".":
            continue
        elif token == "":
            continue
        elif token == "..":
            if len(stack) > 0:
                stack.pop()
        else:
            stack.append(token)
    return "/" + "/".join(stack)

def cd(curr: str, path: str) -> str:
    if path.startswith('/'):
        return simplify(path)
    else:
        return simplify(curr + "/" + path)



class TestCD(unittest.TestCase):

    def setUp(self) -> None:
        self.root = "/home/usr"

    def test_abs_path(self):
        self.assertEqual("/home/sys/fold1", cd(self.root, "/home/sys/fold1"))
    
    def test_relative_path(self):
        self.assertEqual("/home/usr/haoy/testfolder", cd(self.root, "haoy/testfolder"))

        self.assertEqual("/home/usr/haoyc", cd(self.root, "./haoyc"))

        self.assertEqual("/home/haoyc", cd(self.root, "../haoyc"))

        self.assertEqual("/home/usr/haoyc", cd(self.root, "haoyc"))

        self.assertEqual("/haoyc", cd(self.root, "../..//haoyc"))

unittest.main()