import java.math.BigInteger;
class Solution {
    final int MOD = (int)Math.pow(10, 9) + 7;
    public int idealArrays(int n, int maxValue) {
        int dpN = Math.min(n, 14);
        long[][] dp = new long[dpN + 1][maxValue + 1];
        List<List<Integer>> dividends = new ArrayList<>();
        dividends.add(new ArrayList<>());
        for (int i = 1; i <= maxValue; i++) {
            dividends.add(new ArrayList<>());
            int mul = 2;
            while (mul * i <= maxValue) {
                dividends.get(i).add(mul * i);
                mul++;
            }
        }
        Arrays.fill(dp[1], 1);
        dp[1][0] = 0;
        for (int i = 2; i < dp.length; i++) {
            for (int j = 1; j <= maxValue; j++) {
                for (int d : dividends.get(j)) {
                    dp[i][d] += dp[i - 1][j];
                    dp[i][d] %= MOD;
                }
            }
        }
        long res = 0;
        for (int i = 1; i < dp.length; i++) {
            long sum = 0;
            for (long val : dp[i]) {
                sum += val;
                sum %= MOD;
            }
            long tmp = calCombination(n - 1, i - 1) % MOD;
            res += tmp * sum;
            res %= MOD;
        }

        return (int)res;
    }

    private long calCombination(long pos, int delimiter) {
        BigInteger res = BigInteger.ONE;
        BigInteger divisor = BigInteger.ONE;
        for (int i = 0; i < delimiter; i++) {
            res = res.multiply(BigInteger.valueOf(pos - i));
            divisor = divisor.multiply(BigInteger.valueOf(i + 1));
        }
        return res.divide(divisor).mod(BigInteger.valueOf(MOD)).longValue();
    }
}