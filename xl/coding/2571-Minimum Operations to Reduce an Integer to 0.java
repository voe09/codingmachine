// O(log N)
class Solution {
    public int minOperations(int n) {
        int res = 0;
        while (n > 0) {
            if ((n & 0b11) == 0b11) {
                n++;
                res++;
            } else {
            // 01, 10, 00
                if ((n & 1) == 1) {
                    res++;
                }
                n >>= 1;
            }
        }
        return res;
    }
}

// O(N)
class Solution1 {
    public int minOperations(int n) {
        int[] dp = new int[n + 1];
        int power = 0;
        for (int i = 1; i <= n; i++) {
            if ((int)Math.pow(2, power) == i) {
                dp[i] = 1;
                power++;
            } else {
                dp[i] = Integer.MAX_VALUE;
                for (int j = 0; j <= power; j++) {
                    int delta = Math.abs(i - (int)Math.pow(2, j));
                    dp[i] = Math.min(dp[i], dp[delta] + 1);
                }
            }
        }
        return dp[n];
    }
}