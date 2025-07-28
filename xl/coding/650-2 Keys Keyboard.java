class Solution {
    public int minSteps(int n) {
        if (n == 1) {
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = i;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = i + i; j <= n; j += i) {
                dp[j] = Math.min(dp[j], dp[i] + j / i);
            }
        }
        return dp[n];
    }
}