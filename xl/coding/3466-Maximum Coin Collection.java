class Solution {
    public long maxCoins(int[] lane1, int[] lane2) {
        // len > 0
        int len = lane1.length;
        long res = Math.max(lane1[0], lane2[0]);
        // dp1[i][0] - no switches
        // dp1[i][1] - switched twice
        // dp2[i] - switched once
        long[][] dp1 = new long[len][2];
        long[] dp2 = new long[len];
        dp1[0][0] = lane1[0];
        dp1[0][1] = lane1[0];
        dp2[0] = lane2[0];
        for (int i = 1; i < len; i++) {
            dp1[i][0] = Math.max(0, dp1[i - 1][0]) + lane1[i];
            dp1[i][1] = Math.max(0, Math.max(dp2[i - 1], dp1[i - 1][1])) + lane1[i];
            dp2[i] = Math.max(0, Math.max(dp1[i - 1][0], dp2[i - 1])) + lane2[i];
            res = Math.max(res, Math.max(dp1[i][0], Math.max(dp1[i][1], dp2[i])));
        }

        return res;
    }
}