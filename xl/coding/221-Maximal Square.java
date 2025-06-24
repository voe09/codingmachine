class Solution {
    public int maximalSquare(char[][] matrix) {
        // rows, cols > 0
        int rows = matrix.length, cols = matrix[0].length;
        // add one extra row & col to skip checking if i or j is 0
        int[][] dp = new int[rows + 1][cols + 1];
        int maxLen = 0;
        for (int i = 1; i <= rows;i ++) {
            for (int j = 1; j <= cols; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                    maxLen = Math.max(maxLen, dp[i][j]);
                }
            }
        }
        return maxLen * maxLen;
    }
}