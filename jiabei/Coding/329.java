class Solution {
    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    public int longestIncreasingPath(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m][n];
        int max = 0;
        for (int i = 0; i < m; i ++) {
            for (int j = 0; j < n; j ++) {
                max = Math.max(max, dfs(matrix, i, j, m, n, dp));
            }
        }
        return max;
    }

    private int dfs(int[][] matrix, int i, int j, int m, int n, int[][] dp) {
        if (dp[i][j] != 0) {
            return dp[i][j];
        }
        int result = 1;
        for (int k = 0; k < 4; k ++) {
            int ni = i + directions[k][0];
            int nj = j + directions[k][1];
            if (ni >= 0 && ni < m && nj >= 0 && nj < n && matrix[ni][nj] > matrix[i][j]) {
                result = Math.max(result, dfs(matrix, ni, nj, m, n, dp) + 1);
            }
        }
        dp[i][j] = result;
        return dp[i][j];
    }
}