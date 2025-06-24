class Solution {
    public boolean find132pattern(int[] nums) {
        int len = nums.length;
        int[] min = new int[len];
        min[0] = nums[0];
        for (int i = 1; i < len; i++) {
            min[i] = Math.min(min[i - 1], nums[i]);
        }
        Deque<Integer> stack = new ArrayDeque<>();
        for (int k = len - 1; k >= 2; k--) {
            int val = nums[k];
            if (val > min[k - 1] && val < nums[k - 1]) {
                return true;
            }
            while (!stack.isEmpty() && stack.peek() <= min[k - 1]) {
                stack.pop();
            }
            if (!stack.isEmpty() && stack.peek() < nums[k - 1]) {
                return true;
            }
            if (val > min[k - 1]) {
                stack.push(val);
            }
        }
        return false;
    }
}