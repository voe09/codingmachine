class Solution {
    public int longestRepeatingSubstring(String s) {
        int len = s.length();
        // len + 1，方便处理 0行 和 0列
        int[][] dp = new int[len + 1][len + 1];
        int res = 0;
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                    res = Math.max(res, dp[i + 1][j + 1]);
                }
            }
        }
        return res;
    }
}