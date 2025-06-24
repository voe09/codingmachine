class Solution {
    public int climbStairs(int n) {
        // n > 0
        int next1 = 1, next2 = 1;
        int cur = 1;
        for (int i = n + 1 - 3; i >= 0; i--) {
            cur = next1 + next2;
            next2 = next1;
            next1 = cur;
        }
        return cur;
    }
}