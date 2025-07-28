class Solution {
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int len = nums.length;
        int sum = 0;
        for (int i = 0; i < k; i++) {
            sum += nums[i];
        }
        int[] suffixSum = new int[len - k + 1];
        suffixSum[0] = sum;
        for (int i = 1; i <= len - k; i++) {
            sum -= nums[i - 1];
            sum += nums[i - 1 + k];
            suffixSum[i] = sum;
        }
        int[][] maxSum = new int[len - k + 1][2];
        maxSum[len - k][0] = suffixSum[len - k];
        maxSum[len - k][1] = len - k;
        for (int i = len - k - 1; i >= 0; i--) {
            if (suffixSum[i] >= maxSum[i + 1][0]) {
                maxSum[i][0] = suffixSum[i];
                maxSum[i][1] = i;
            } else {
                maxSum[i][0] = maxSum[i + 1][0];
                maxSum[i][1] = maxSum[i + 1][1];
            }
        }
        // for 2 subarrays
        int[][] maxSumTwoArrays = new int[len - k - k + 1][3];
        maxSumTwoArrays[len - k - k][0] = suffixSum[len - k - k] + suffixSum[len - k];
        maxSumTwoArrays[len - k - k][1] = len - k - k;
        maxSumTwoArrays[len - k - k][2] = len - k;
        for (int i = len - k - k - 1; i >= 0; i--) {
            int curMaxTwo = suffixSum[i] + maxSum[i + k][0];
            if (curMaxTwo >= maxSumTwoArrays[i + 1][0]) {
                maxSumTwoArrays[i][0] = curMaxTwo;
                maxSumTwoArrays[i][1] = i;
                maxSumTwoArrays[i][2] = maxSum[i + k][1];
            } else {
                maxSumTwoArrays[i][0] = maxSumTwoArrays[i + 1][0];
                maxSumTwoArrays[i][1] = maxSumTwoArrays[i + 1][1];
                maxSumTwoArrays[i][2] = maxSumTwoArrays[i + 1][2];
            }
        }

        // for 3 subarrays
        int finalMax = 0;
        int[] res = new int[3];
        for (int i = 0; i <= len - 3 * k; i++) {
            if (suffixSum[i] + maxSumTwoArrays[i + k][0] > finalMax) {
                finalMax = suffixSum[i] + maxSumTwoArrays[i + k][0];
                res[0] = i;
                res[1] = maxSumTwoArrays[i + k][1];
                res[2] = maxSumTwoArrays[i + k][2];
            }
        }
        return res;
    }
}