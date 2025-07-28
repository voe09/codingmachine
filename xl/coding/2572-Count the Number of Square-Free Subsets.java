class Solution {
    private long[][] mem;
    private int[] decToBi;
    private final int MOD = (int)Math.pow(10, 9) + 7;
    public int squareFreeSubsets(int[] nums) {
        // 10 prime numbers
        int[] primes = new int[] {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        decToBi = new int[31];
        for (int i = 0; i < primes.length; i++) {
            decToBi[primes[i]] = 1 << i;
        }
        for (int i = 2; i < decToBi.length; i++) {
            if (decToBi[i] != 0) {
                continue;
            }
            int mask = 0;
            for (int prime : primes) {
                if (prime >= i) {
                    break;
                }
                if (i % prime == 0) {
                    mask |= decToBi[prime];
                    if (i / prime % prime == 0) {
                        mask = -1;
                        break;
                    }
                }
            }
            decToBi[i] = mask;
        }

        int len = nums.length;
        mem = new long[len][1 << 10];
        for (long[] prime : mem) {
            Arrays.fill(prime, -1);
        }
        return ((int)dfs(nums, 0, 0) - 1 + MOD) % MOD;
    }

    private long dfs(int[] nums, int start, int mask) {
        if (start == nums.length) {
            return 1;
        }
        if (mem[start][mask] != -1) {
            return mem[start][mask];
        }
        int num = nums[start];
        long res = dfs(nums, start + 1, mask) % MOD;
        if (decToBi[num] != -1 && (mask & decToBi[num]) == 0) {
            res += dfs(nums, start + 1, mask | decToBi[num]);
            res %= MOD;
        }
        mem[start][mask] = res;
        return res;
    }
}