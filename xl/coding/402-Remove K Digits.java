class Solution {
    public String removeKdigits(String num, int k) {
        Deque<Character> dq = new ArrayDeque<Character>();
        for (char c : num.toCharArray()) {
            while (k > 0 && !dq.isEmpty() && dq.peek() > c) {
                dq.pop();
                k--;
            }
            if (dq.isEmpty() && c == '0') {
                continue;
            }
            dq.push(c);
        }
        while (k > 0 && !dq.isEmpty()) {
            dq.pop();
            k--;
        }
        if (dq.isEmpty()) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        while (!dq.isEmpty()) {
            sb.append(dq.pollLast());
        }
        return sb.toString();
    }
}