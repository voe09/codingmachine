class Solution {
    public int sumSubarrayMins(int[] arr) {
        final int MODULO = (int)Math.pow(10, 9) + 7;
        int len = arr.length;
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(-1);
        long sum = 0;
        for (int i = 0; i < len; i++) {
            int cur = arr[i];
            while (stack.size() > 1 && arr[stack.peek()] >= cur) {
                int lastIdx = stack.pop();
                int leftBound = stack.peek(), rightBound = i;
                int min = arr[lastIdx];
                sum += (long)min * (lastIdx - leftBound) * (rightBound - lastIdx);
                sum %= MODULO;
            }
            stack.push(i);
        }
        while (stack.size() > 1) {
            int lastIdx = stack.pop();
            int leftBound = stack.peek(), rightBound = len;
            int min = arr[lastIdx];
            sum += (long)min * (lastIdx - leftBound) * (rightBound - lastIdx);
            sum %= MODULO;
        }
        return (int)sum;
    }
}