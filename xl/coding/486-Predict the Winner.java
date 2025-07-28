class Solution {
    private int[][] mem;
    public boolean predictTheWinner(int[] nums) {
        int len = nums.length;
        mem = new int[len][len];
        for (int[] row : mem) {
            Arrays.fill(row, Integer.MIN_VALUE);
        }
        return maxDiff(nums, 0, len - 1) >= 0;
    }

    private int maxDiff(int[] nums, int start, int end) {
        if (start == end) {
            return nums[start];
        }
        if (mem[start][end] != Integer.MIN_VALUE) {
            return mem[start][end];
        }

        mem[start][end] = Math.max(nums[start] - maxDiff(nums, start + 1, end), nums[end] - maxDiff(nums, start, end - 1));
        return mem[start][end];
    }
}