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





class TreeNode:

    def __init__(self, token: str):
        self.token = token
        self.children = {}
        self.replace = None

    def add(self, path: str, replace: str):
        tokens = path.split("/")[1:]
        curr = self
        for i in range(len(tokens)):
            token = tokens[i]
            if token not in curr.children:
                curr.children[token] = TreeNode(token)
            curr = curr.children[token]
            if i == len(tokens) - 1:
                curr.replace = replace

    def visit(self, path: str) -> tuple[str, bool]:
        tokens = path.split("/")[1:]

        curr = self
        updated = False
        pos = -1
        new = None
        for i in range(len(tokens)):
            token = tokens[i]
            if token not in curr.children:
                break
            curr = curr.children[token]
            if curr.replace is not None:
                pos = i
                new = curr.replace
                updated = True
        
        if updated:
            new_path = path.replace("/" + "/".join(tokens[:pos+1]), new)
            return (new_path, True)
        return (path, False)


def cd_with_symlink(curr: str, path: str, symlinks: dict[str, str]) -> str:
    if not path.startswith("/"):
        path = curr + "/" + path

    root = TreeNode("")
    for origin, replace in symlinks.items():
        root.add(simplify(origin), simplify(replace))

    path = simplify(path)
    visited = set([path])
    while True:
        path, updated = root.visit(path)
        if not updated:
            break
    
        path = simplify(path)
        if path in visited:
            raise ValueError("Circle Detected")
        visited.add(path)
    
    return path

class TestCDWithSymlink(unittest.TestCase):

    def setUp(self) -> None:
        self.root = "/home/usr"
    
    def test_replace_once_successfully(self):
        self.assertEqual("/usr/circle", 
        cd_with_symlink(self.root, "haoyc/circle", {self.root + "/haoyc": "/usr"}))

        # longer one is being taken
        self.assertEqual("/usr/circle", 
        cd_with_symlink(self.root, "haoyc/circle", {self.root + "/haoyc": "/usr", self.root: "/dummy"}))

    def test_replace_3_times_successfully(self):
        self.assertEqual("/app/circle", 
        cd_with_symlink(self.root, "haoyc/circle", {
            self.root + "/haoyc": "/usr", 
            self.root: "/dummy",
            "/usr/circle": "/usr/haoyc/app/circle",
            "//usr/haoyc/app": "/app"
        }))

    def test_circular_dependency(self):
        with self.assertRaises(ValueError):
            cd_with_symlink(self.root, "haoyc/circle", {
            self.root + "/haoyc": "/usr", 
            self.root: "/dummy",
            "/usr/circle": "/home/usr/haoyc/circle",
            "//usr/haoyc/app": "/app"
        })

unittest.main()
