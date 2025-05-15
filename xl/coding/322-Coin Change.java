// O(NS)
class Solution {
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for (int i = 1; i < dp.length; i++) {
            for (int coin : coins) {
                if (coin > i) {
                    continue;
                }
                int delta = i - coin;
                if (dp[delta] >= 0) {
                    dp[i] = dp[i] == -1 ? dp[delta] + 1 : Math.min(dp[i], dp[delta] + 1);
                }
            }
        }
        return dp[amount];
    }
}