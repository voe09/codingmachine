// time - O(N)
// space - O(1) if just keep the prev
class Solution {
    public int maxProduct(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        int[][] dp = new int[len][2];
        if (nums[0] > 0) {
            dp[0][0] = nums[0];
        } else {
            dp[0][1] = nums[0];
        }
        int res = nums[0];
        for (int i = 1; i < len; i++) {
            if (nums[i] > 0) {
                dp[i][0] = dp[i - 1][0] != 0 ? dp[i - 1][0] * nums[i] : nums[i];
                dp[i][1] = dp[i - 1][1] != 0 ? dp[i - 1][1] * nums[i] : 0;
            } else {
                dp[i][0] = dp[i - 1][1] != 0 ? dp[i - 1][1] * nums[i] : 0;
                dp[i][1] = dp[i - 1][0] != 0 ? dp[i - 1][0] * nums[i] : nums[i];
            }
            res = Math.max(res, dp[i][0]);
            // res = Math.max(res, dp[i][1]);
        }
        return res;
    }
}