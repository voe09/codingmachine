class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        // len > 0, nums[i] >= 0
        int len = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (Math.abs(target) > sum) {
            return 0;
        }
        int[][] mem = new int[len][sum * 2 + 1];
        for (int[] row : mem) {
            Arrays.fill(row, -1);
        }
        return countWays(nums, mem, 0, 0, target, sum);
    }

    private int countWays(int[] nums, int[][] mem, int start, int curSum, int target, int total) {
        if (mem[start][curSum + total] >= 0) {
            return mem[start][curSum + total];
        }
        mem[start][curSum + total] = 0;
        if (start == nums.length - 1) {
            if (curSum + nums[start] == target) {
                mem[start][curSum + total]++;
            }
            if (curSum - nums[start] == target) {
                mem[start][curSum + total]++;
            }
            return mem[start][curSum + total];
        }
        mem[start][curSum + total] += countWays(nums, mem, start + 1, curSum + nums[start], target, total);
        mem[start][curSum + total] += countWays(nums, mem, start + 1, curSum - nums[start], target, total);
        return mem[start][curSum + total];
    }
}

class Solution1 {
    public int findTargetSumWays(int[] nums, int target) {
        // len > 0, nums[i] >= 0
        int len = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (Math.abs(target) > sum) {
            return 0;
        }
        int[][] dp = new int[len][sum * 2 + 1];
        dp[0][sum - nums[0]] += 1;
        dp[0][sum + nums[0]] += 1;
        for (int i = 1; i < len; i++) {
            int cur = nums[i];
            for (int j = 0; j < dp[i - 1].length; j++) {
                if (dp[i - 1][j] != 0) {
                    dp[i][j + cur] += dp[i - 1][j];
                    dp[i][j - cur] += dp[i - 1][j];
                }
            }
        }
        return dp[len - 1][target + sum];
    }
}