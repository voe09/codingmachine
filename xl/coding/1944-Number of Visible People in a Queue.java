class Solution {
    public int[] canSeePersonsCount(int[] heights) {
        int len = heights.length;
        Deque<Integer> stack = new ArrayDeque<>();
        int[] res = new int[len];
        for (int i = len - 1; i >= 0; i--) {
            int count = 0;
            int cur = heights[i];
            while (!stack.isEmpty() && stack.peek() < cur) {
                stack.pop();
                count++;
            }
            if (!stack.isEmpty()) {
                count++;
            }
            stack.push(cur);
            res[i] = count;
        }
        return res;
    }
}