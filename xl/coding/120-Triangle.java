class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        // len > 0
        int len = triangle.size();
        int[] dp = new int[triangle.get(len - 1).size()];
        // last row
        for (int i = 0; i < dp.length; i++) {
            dp[i] = triangle.get(len - 1).get(i);
        }
        // dp
        for (int i = len - 2; i >= 0; i--) {
            int[] curDp = new int[i + 1];
            for (int j = 0; j < curDp.length; j++) {
                curDp[j] = triangle.get(i).get(j) + Math.min(dp[j], dp[j + 1]);
            }
            dp = curDp;
        }
        return dp[0];
    }
}