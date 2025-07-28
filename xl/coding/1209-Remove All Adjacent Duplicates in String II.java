class Solution {
    private class Node {
        char c;
        int count;

        public Node(char c, int count) {
            this.c = c;
            this.count = count;
        }
    }
    public String removeDuplicates(String s, int k) {
        int len = s.length();
        Deque<Node> dq = new ArrayDeque<>();
        char prev = s.charAt(0);
        int prevCount = 1;
        for (int i = 1; i < len; i++) {
            char cur = s.charAt(i);
            if (cur == prev) {
                prevCount++;
            } else {
                prevCount %= k;
                if (prevCount != 0) {
                    if (dq.isEmpty() || dq.peek().c != prev) {
                        dq.push(new Node(prev, prevCount));
                    } else if ((dq.peek().count + prevCount) % k != 0) {
                        dq.peek().count = (dq.peek().count + prevCount) % k;
                    } else if ((dq.peek().count + prevCount) % k == 0) {
                        dq.pop();
                    }
                }
                prev = cur;
                prevCount = 1;
            }
        }
        prevCount %= k;
        if (prevCount != 0) {
            if (dq.isEmpty() || dq.peek().c != prev) {
                dq.push(new Node(prev, prevCount));
            } else if ((dq.peek().count + prevCount) % k != 0) {
                dq.peek().count = (dq.peek().count + prevCount) % k;
            } else if ((dq.peek().count + prevCount) % k == 0) {
                dq.pop();
            }
        }
        StringBuilder res = new StringBuilder();
        while (!dq.isEmpty()) {
            Node cur = dq.pollLast();
            for (int i = 0; i < cur.count; i++) {
                res.append(cur.c);
            }
        }
        return res.toString();
    }
}