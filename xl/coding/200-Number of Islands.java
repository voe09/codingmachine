// dfs
// O(M * N)
class Solution {
    private int[][] dirs = new int[][] {
        {0, -1},
        {0,  1},
        { 1, 0},
        {-1, 0}
    };
    public int numIslands(char[][] grid) {
        int res = 0;
        int rows = grid.length;
        if (rows == 0) {
            return res;
        }
        int cols = grid[0].length;
        if (cols == 0) {
            return res;
        }
        boolean[][] visited = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1' && !visited[i][j]) {
                    res++;
                    dfs(grid, visited, i, j);
                }
            }
        }
        return  res;
    }

    private void dfs(char[][] grid, boolean[][] visited, int r, int c) {
        visited[r][c] = true;
        for (int[] dir : dirs) {
            int newR = r + dir[0];
            int newC = c + dir[1];
            if (newR < 0 || newR >= grid.length || newC < 0 || newC >= grid[0].length 
                || visited[newR][newC] || grid[newR][newC] == '0') {
                continue;
            }
            dfs(grid, visited, newR, newC);
        }
    }
}

///////////////////////////////////////
// union find
// O(M * N)

class Solution {
    private int[][] dirs = new int[][] {
        {0, -1},
        {0,  1},
        { 1, 0},
        {-1, 0}
    };

    class UnionFind {
        int[] parent;
        int[] size;

        public UnionFind(char[][] grid) {
            int rows = grid.length;
            int cols = grid[0].length;
            parent = new int[rows * cols];
            size = new int[rows * cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    parent[i * cols + j] = i * cols + j;
                    size[i * cols + j] = 1;
                }
            }
        }

        public int find(int x) {
            if (parent[x] == x) {
                return x;
            }
            parent[x] = find(parent[x]);
            return parent[x];
        }

        public boolean union(int a, int b) {
            int parentA = find(a);
            int parentB = find(b);
            if (parentA == parentB) {
                return false;
            }
            if (size[parentA] > size[b]) {
                parent[parentB] = parentA;
                size[parentA] += size[parentB];
            } else {
                parent[parentA] = parentB;
                size[parentB] += size[parentA];
            }
            return true;
        }
    }


    public int numIslands(char[][] grid) {
        int res = 0;
        int rows = grid.length;
        if (rows == 0) {
            return res;
        }
        int cols = grid[0].length;
        if (cols == 0) {
            return res;
        }
        res = rows * cols;
        UnionFind uf = new UnionFind(grid);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '0') {
                    res--;
                } else {
                    for (int[] dir : dirs) {
                        int newR = i + dir[0];
                        int newC = j + dir[1];
                        if (newR < 0 || newR >= rows || newC < 0 || newC >= cols 
                            || grid[newR][newC] == '0') {
                            continue;
                        }
                        if (uf.union(i * cols + j, newR * cols + newC)) {
                            res--;
                        }
                    }
                }
            }
        }
        return  res;
    }
}