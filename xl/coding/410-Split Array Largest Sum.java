// O(N log maxSum)
class Solution {
    public int splitArray(int[] nums, int k) {
        int end = 1000 * (int)Math.pow(10, 6);
        int start = 0;
        int mid = 0;
        while (start < end) {
            mid = (end - start) / 2 + start;
            if (isValid(nums, k, mid)) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return start;
    }
    
    private boolean isValid(int[] nums, int k, int sum) {
        int curSum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > sum) {
                return false;
            }
            curSum += nums[i];
            if (curSum > sum) {
                curSum = nums[i];
                k--;
                if (k == 0) {
                    return false;
                }
            }
        }
        return true;
    }
}

// O(N ^ 2 * K)
class Solution1 {
    private int res = Integer.MAX_VALUE;
    public int splitArray(int[] nums, int k) {
        int len = nums.length;
        int[][] mem = new int[len][k];
        int[] suffixSum = new int[len];
        suffixSum[len - 1] = nums[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            suffixSum[i] = nums[i] + suffixSum[i + 1];
        }

        getMinLargest(nums, suffixSum, 0, k - 1, 0, mem);
        return mem[0][k - 1];
    }

    private int getMinLargest(int[] nums, int[] suffixSum, int start, int groups, int curLargestSubSum, int[][] mem) {
        if (mem[start][groups] != 0) {
            return mem[start][groups];
        }
        if (groups == 0) {
            mem[start][groups] = suffixSum[start];
            return mem[start][groups];
        }
        mem[start][groups] = Integer.MAX_VALUE;
        int curSum = 0;
        for (int i = start; i < nums.length - groups; i++) {
            curSum += nums[i];
            int tmp = getMinLargest(nums, suffixSum, i + 1, groups - 1, Math.max(curSum, curLargestSubSum), mem);
            mem[start][groups] = Math.min(mem[start][groups], Math.max(curSum, tmp));
            if (curSum >= mem[start][groups]) {
                break;
            }
        }
        return mem[start][groups];
    }
}