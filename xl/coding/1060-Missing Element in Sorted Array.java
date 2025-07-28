class Solution {
    public int missingElement(int[] nums, int k) {
        int start = 0, end = nums.length - 1;
        int mid = 0;
        while (start <= end) {
            mid = (end - start) / 2 + start;
            if (k <= nums[mid] - (mid + nums[0])) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        // k >= 1, so end >= 0
        return nums[end] + k - (nums[end] - (end + nums[0]));
    }
}