class Solution {
    public int findLength(int[] nums1, int[] nums2) {
        int len1 = nums1.length, len2 = nums2.length;
        int[][] dp = new int[len1][len2];
        int res = 0;
        for (int i = 0; i < len2; i++) {
            if (nums1[0] == nums2[i]) {
                dp[0][i] = 1;
                res = 1;
            }
        }
        for (int i = 0; i < len1; i++) {
            if (nums1[i] == nums2[0]) {
                dp[i][0] = 1;
                res = 1;
            }
        }
        for (int i = 1; i < len1; i++) {
            for (int j = 1; j < len2; j++) {
                if (nums1[i] == nums2[j]) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                    res = Math.max(res, dp[i][j]);
                }
            }
        }
        return res;
    }
}