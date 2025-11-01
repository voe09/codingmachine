from heapq import heappush, heappop

def shorted_path(grid: list[list[str]]):
    # BFS build shorted path between points
    # dict: (x, y) -> (x, y), path
    m, n = len(grid), len(grid[0])
    graph = {}
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '*':
                path = []
                visited = set([(i, j)])
                graph[(i, j)] = {}
                dfs(grid, (i, j), i, j, visited, path, graph)

    # dijkstra algorithm
    path = []
    node = next(iter(graph))
    visited = set()
    heap = [(0, node, [node])] # list of tuple, 1st is len, second is the node, 3rd is the path

    while len(heap) > 0:
        _, node, to_path = heappop(heap)
        if node not in visited:
            path.extend(to_path)
            visited.add(node)
            for neighbor, neighbor_path in graph[node].items():
                heappush(heap, (len(neighbor_path), neighbor, neighbor_path))

    if len(visited) != len(graph):
        return []

    return path


dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def dfs(grid: list[list[str]], origin: tuple, i: int, j: int, visited: set, path: list, graph: dict):
    m, n = len(grid), len(grid[0])
    for dir in dirs:
        x = i + dir[0]
        y = j + dir[1]
        if not (0 <= x < m and 0 <= y < n):
            continue
        if (x, y) in visited:
            continue
        if grid[x][y] == '#':
            continue
        path.append((x, y))
        visited.add((x, y))
        if grid[x][y] == '*':
            if (x, y) not in graph[origin]:
                graph[origin][(x, y)] = list(path)
            else:
                if len(graph[origin][(x, y)]) > len(path):
                    graph[origin][(x, y)] = list(path)
        else:
            dfs(grid, origin, x, y, visited, path, graph)
        visited.remove((x, y))
        path.pop()



grid = [
    [".", "*", "."],
    ["*", ".", "*"],
    ["*", ".", "*"],
]

print(shorted_path(grid))