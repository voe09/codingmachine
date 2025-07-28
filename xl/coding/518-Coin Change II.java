class Solution {
    private int[][] mem;

    public int change(int amount, int[] coins) {
        mem = new int[amount + 1][coins.length];
        for (int[] row : mem) {
            Arrays.fill(row, -1);
        }
        return getCombinations(amount, coins, 0);
    }

    private int getCombinations(int amount, int[] coins, int cur) {
        if (amount == 0) {
            return  1;
        }
        if (amount < 0 || cur == coins.length) {
            return 0;
        }
        if (mem[amount][cur] != -1) {
            return mem[amount][cur];
        }
        int res = getCombinations(amount - coins[cur], coins, cur);
        res += getCombinations(amount, coins, cur + 1);
        
        mem[amount][cur] = res;
        return mem[amount][cur];
    }
}