class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int len = nums.length;
        int[] res = new int[len];
        Arrays.fill(res, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        getGreater(nums, stack, res);
        getGreater(nums, stack, res);
        return res;
    }

    private void getGreater(int[] nums, Deque<Integer> stack, int[] res) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            int cur = nums[i];
            while (!stack.isEmpty() && nums[stack.peek()] < cur) {
                res[stack.pop()] = cur;
            }
            stack.push(i);
        }
    }
}