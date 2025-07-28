class Solution {
    int[] mem;
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int n = nums.length;
        mem = new int[1 << n];
        int total = 0;
        for (int num : nums) {
            total += num;
        }
        if (total % k != 0) {
            return false;
        }
        return backtracking(nums, k, 0, 0, total / k);
    }

    private boolean backtracking(int[] nums, int k, int mask, int sum, int target) {
        if (mem[mask] != 0) {
            return mem[mask] == 1;
        }
        if (k == 0) {
            return true;
        }
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int shift = 1 << i;
            int nextSum = sum + nums[i];
            if ((mask & shift) == 0 && nextSum <= target) {
                if (backtracking(nums, nextSum == target ? k - 1 : k, mask | shift, nextSum == target ? 0 : nextSum, target)) {
                    return true;
                }
            }
        }
        mem[mask] = -1;
        return false;
    }
}