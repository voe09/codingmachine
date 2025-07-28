class Solution {
    public int countSquares(int[][] matrix) {
        // rows, cols > 0
        int rows = matrix.length, cols = matrix[0].length;
        int[][] dp = new int[rows][cols];
        int res = 0;
        for (int i = 0; i < rows; i++) {
            if (matrix[i][0] == 1) {
                dp[i][0] = 1;
                res++;
            }
        }
        for (int j = 0; j < cols; j++) {
            if (matrix[0][j] == 1) {
                dp[0][j] = 1;
                res++;
            }
        }
        res -= dp[0][0];
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    continue;
                }
                dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                res += dp[i][j];
            }
        }
        return res;
    }
}