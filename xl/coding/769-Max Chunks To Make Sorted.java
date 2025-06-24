class Solution {
    public int maxChunksToSorted(int[] arr) {
        int len = arr.length;
        int res = 1;
        int end = 0;
        for (int i = 0; i < len; i++) {
            if (i > end) {
                end = arr[i];
                res++;
            } else if (arr[i] > end) {
                end = arr[i];
            }
        }
        return res;
    }
}

// monotonic stack
class Solution1 {
    public int maxChunksToSorted(int[] arr) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (int val : arr) {
            if (stack.isEmpty() || stack.peek() < val) {
                stack.push(val);
            } else {
                int max = stack.pop();
                while (!stack.isEmpty() && stack.peek() > val) {
                    stack.pop();
                }
                stack.push(max);
            }
        }
        return stack.size();
    }
}