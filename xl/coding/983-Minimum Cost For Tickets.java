class Solution {
    public int mincostTickets(int[] days, int[] costs) {
        // days 365, strictly increasing, costs > 0
        int len = days.length;
        int[] dp = new int[days[len - 1] + 1];
        int idx = 0;
        for (int cur = days[0]; cur <= days[len - 1]; cur++) {
            if (cur < days[idx]) {
                dp[cur] = dp[cur - 1];
                continue;
            }
            int preCost0 = getPreCost(dp, cur - 1, days[0]);
            int preCost1 = getPreCost(dp, cur - 7, days[0]);
            int preCost2 = getPreCost(dp, cur - 30, days[0]);
            int min = Math.min(preCost0 + costs[0], Math.min(preCost1 + costs[1], preCost2 + costs[2]));
            dp[cur] = min;
            idx++;
        }
        return dp[days[len - 1]];
    }

    private int getPreCost(int[] dp, int day, int first) {
        if (day < first) {
            return 0;
        }
        return dp[day];
    }
}