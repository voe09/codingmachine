class Solution {
    public int rob(int[] nums) {
        // len > 0
        int len = nums.length;
        if (len == 1) {
            return nums[0];
        }
        // 0 - rob cur, rob 0
        // 1 - rob cur, not rob 0
        // 2 - not rob cur, rob 0
        // 3 - not rob cur, not rob 0
        int[][] dp = new int[len][4];
        dp[0] = new int[] { nums[0], 0, 0, 0 };
        for (int i = 1; i < len; i++) {
            dp[i][0] = dp[i - 1][2] + nums[i];
            dp[i][1] = dp[i - 1][3] + nums[i];
            dp[i][2] = Math.max(dp[i - 1][0], dp[i - 1][2]);
            dp[i][3] = Math.max(dp[i - 1][1], dp[i - 1][3]);
        }
        return Math.max(dp[len - 1][1], Math.max(dp[len - 1][2], dp[len - 1][3]));
    }
}