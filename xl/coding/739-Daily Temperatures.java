class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int len = temperatures.length;
        int[] res = new int[len];
        // 单调递减
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < len; i++) {
            int cur = temperatures[i];
            while (!stack.isEmpty() && temperatures[stack.peek()] < cur) {
                int toSet = stack.pop();
                res[toSet] = i - toSet;
            }
            stack.push(i);
        }
        return res;
    }
}