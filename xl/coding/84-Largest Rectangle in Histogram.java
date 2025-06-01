// O(N), monotonic stack
// 也可以 divide and conquer，O(N log N)，（用segment tree优化）
class Solution {
    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stack = new ArrayDeque<Integer>();
        int len = heights.length;
        int res = 0;
        // rect from start
        stack.push(-1);
        for (int i = 0; i < len; i++) {
            int cur = heights[i];
            while (stack.size() > 1 && heights[stack.peek()] > cur) {
                int prev = heights[stack.pop()];
                res = Math.max(res, (i - 1 - stack.peek()) * prev);
            }
            stack.push(i);
        }
        while (stack.size() > 1) {
            int prev = heights[stack.pop()];
            res = Math.max(res, (len - 1 - stack.peek()) * prev);
        }
        return res;
    }
}