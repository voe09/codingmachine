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


def cd_with_symlinks(
    current_dir: str, 
    new_dir: str, 
    symlinks: dict[str, str],
    home: str = "/home/usr",
) -> str:
    path = cd(current_dir, new_dir, home)
    if path == "NULL":
        return path

    visited = set()
    while True:
        if path in visited:
            raise ValueError("circular dep")

        visited.add(path)

        match = None
        for k in symlinks:
            if path == k or path.startswith(k + "/"):
                if match is None or len(k) > len(match):
                    match = k

        if match is None:
            break
        
        target = symlinks[match]
        path = path.replace(match, target, 1)

        path = simplify_path(path)
        if path == "NULL":
            return path
    
    return path

print("test symlinks")
print(cd_with_symlinks("/foo/bar", "baz", {"/foo/bar": "/abc", "/foo/bar/baz": "/xyz"}))