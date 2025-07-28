class Solution {
    public int maxProfit(int k, int[] prices) {
        int len = prices.length;
        int[][][] dp = new int[len][k + 1][2];
        // 0 - holding a stock
        // 1 - not holding
        dp[0][1][0] = -prices[0];
        for (int i = 1; i < len; i++) {
            for (int j = 1; j <= k; j++) {
                if ((j - 1) * 2 == i) {
                    dp[i][j][0] = dp[i - 1][j - 1][1] - prices[i];
                } else if ((j - 1) * 2 < i) {
                    dp[i][j][0] = Math.max(dp[i - 1][j - 1][1] - prices[i], dp[i - 1][j][0]);
                }
                if ((j - 1) * 2 + 1 <= i) {
                    dp[i][j][1] = Math.max(dp[i - 1][j][0] + prices[i], dp[i - 1][j][1]);
                }
            }
        }
        int res = 0;
        for (int[] profit : dp[len - 1]) {
            res = Math.max(res, profit[1]);
        }
        return res;
    }
}