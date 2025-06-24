class Solution {
    public int longestValidParentheses(String s) {
        int res = 0;
        int len = s.length();
        int left = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c == '(') {
                left++;
                stack.push(0);
            } else {
                left--;
                if (left < 0) {
                    left = 0;
                    stack.clear();
                } else {
                    int cur = 0;
                    while (stack.peek() > 0) {
                        cur += stack.pop();
                    }
                    stack.pop();
                    cur += 2;
                    while (!stack.isEmpty() && stack.peek() > 0) {
                        cur += stack.pop();
                    }
                    stack.push(cur);
                    res = Math.max(res, cur);
                }
            }
        }
        return res;
    }
}