// O(M * N), DFS + mem

class Solution {
    int[][] dirs = new int[][] {
        { 0, -1},
        { 0,  1},
        { 1,  0},
        {-1,  0}
    };
    public int longestIncreasingPath(int[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        int[][] path = new int[rows][cols];
        int res = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res = Math.max(res, dfs(matrix, path, i, j));
            }
        }
        return res;
    }

    private int dfs(int[][] matrix, int[][] path, int row, int col) {
        if (path[row][col] > 0) {
            return path[row][col];
        }
        path[row][col] = 1;
        int rows = matrix.length, cols = matrix[0].length;
        for (int[] dir : dirs) {
            int newR = row + dir[0];
            int newC = col + dir[1];
            if (newR < 0 || newR >= rows || newC < 0 || newC >= cols 
                || matrix[newR][newC] <= matrix[row][col]) {
                continue;
            }
            int pathLen = dfs(matrix, path, newR, newC);
            path[row][col] = Math.max(path[row][col], pathLen + 1);
        }

        return path[row][col];
    }
}

////////////////////////
// 也可以 topological sort
