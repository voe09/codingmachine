class Solution {
    private int[] dp;

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (desiredTotal == 0) {
            return true;
        }
        if ((1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal) {
            return false;
        }
        int totalCombination = 1 << maxChoosableInteger;
        dp = new int[totalCombination];
        Arrays.fill(dp, -1);
        return getRes(0, 0, maxChoosableInteger, desiredTotal);
    }

    private boolean getRes(int mask, int curSum, int maxChoosableInteger, int desiredTotal) {
        if (curSum >= desiredTotal) {
            return false;
        }
        if (dp[mask] != -1) {
            return dp[mask] == 1;
        }
        for (int shift = 0; shift < maxChoosableInteger; shift++) {
            int cur = 1 << shift;
            if ((mask & cur) == 0) {
                boolean res = getRes(mask | cur, curSum + shift + 1, maxChoosableInteger, desiredTotal);
                if (!res) {
                    dp[mask] = 1;
                    return true;
                }
            }
        }
        dp[mask] = 0;
        return false;
    }
}