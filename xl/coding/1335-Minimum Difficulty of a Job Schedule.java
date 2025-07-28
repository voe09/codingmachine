class Solution {
    public int minDifficulty(int[] jobDifficulty, int d) {
        int len = jobDifficulty.length;
        if (len < d) {
            return -1;
        }
        // dp[i][j] start from i th job, day days left (includes today)
        int[][] dp = new int[len][d + 1];
        dp[len - 1][1] = jobDifficulty[len - 1];
        for (int i = len - 1 - 1; i >= d - 1; i--) {
            dp[i][1] = Math.max(jobDifficulty[i], dp[i + 1][1]);
        }
        for (int day = 2; day <= d; day++) {
            for (int i = len - 1 - (day - 1); i >= d - day; i--) {
                int hardest = 0;
                dp[i][day] = Integer.MAX_VALUE;
                for (int j = i; j <= len - 1 - (day - 1); j++) {
                    hardest = Math.max(hardest, jobDifficulty[j]);
                    dp[i][day] = Math.min(dp[i][day], hardest + dp[j + 1][day - 1]);
                }
            }
        }
        return dp[0][d];
    }
}