class Solution {
    public int minNumberOperations(int[] target) {
        int len = target.length;
        int res = 0;
        Deque<Integer> stack = new ArrayDeque<Integer>();
        // target > 0
        stack.push(0);
        boolean poped = false;
        for (int val : target) {
            while (stack.peek() > val) {
                stack.pop();
                poped = true;
            }
            if (poped) {
                stack.push(val);
                poped = false;
                continue;
            }
            res += val - stack.peek();
            stack.push(val);
        }
        return res;
    }
}