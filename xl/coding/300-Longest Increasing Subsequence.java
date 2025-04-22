// 354. Russian Doll Envelopes 青春版 

// recording the min last number of each specific subsequence length
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        // subEnd[i] -> the min last number for a sub of length i
        int[] subEnd = new int[len];
        int end = 0;
        subEnd[0] = nums[0];
        int res = 1;
        for (int i = 1; i < len; i++) {
            end = findPos(subEnd, 0, end, nums[i]);
            res = Math.max(res, end + 1);
        }

        return res;
    }

    private int findPos(int[] subEnd, int start, int end, int num) {
        int oriEnd = end;
        while (start <= end) {
            int mid = (end - start) / 2 + start;
            int cur = subEnd[mid];
            if (cur < num) {
                start = mid + 1;
            } else if (cur > num) {
                end = mid - 1;
            } else {
                start = mid;
                break;
            }
        }
        subEnd[start] = num;
        return Math.max(oriEnd, start);
    }
}

//////////////////////////////////////////////
// dp, O(N^2)
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        // dp[i] -> LIS len when end with nums[i]
        int[] dp = new int[len];
        dp[0] = 1;
        int res = 1;
        for (int i = 1; i < len; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }

        return res;
    }
}

