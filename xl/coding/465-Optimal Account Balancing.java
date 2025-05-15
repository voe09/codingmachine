// O(N * 2^N)
class Solution {
    public int minTransfers(int[][] transactions) {
        int[] blcs = new int[12];
        for (int[] trx : transactions) {
            blcs[trx[0]] -= trx[2];
            blcs[trx[1]] += trx[2];
        }
        List<Integer> accBlcs = new ArrayList<Integer>();
        for (int blc : blcs) {
            if (blc != 0) {
                accBlcs.add(blc);
            }
        }
        int len = accBlcs.size();
        int[] mem = new int[(1 << len)];
        Arrays.fill(mem, -1);
        int count = maxBalancedSubgroup(accBlcs, mem, (1 << len) - 1);
        return len - count;
    }

    private int maxBalancedSubgroup(List<Integer> accBlcs, int[] mem, int bitmasks) {
        if (bitmasks == 0) {
            return 0;
        }
        if (mem[bitmasks] >= 0) {
            return mem[bitmasks];
        }
        int sum = 0;
        for (int i = 0; i < accBlcs.size(); i++) {
            int mask = 1 << i;
            if ((bitmasks & mask) > 0) {
                sum += accBlcs.get(i);
            }
        }
        int isBalanced = sum == 0 ? 1 : 0;
        for (int i = 0; i < accBlcs.size(); i++) {
            int mask = 1 << i;
            if ((bitmasks & mask) == 0) {
                // no debt for this account
                continue;
            }
            mem[bitmasks] = Math.max(mem[bitmasks], isBalanced + maxBalancedSubgroup(accBlcs, mem, bitmasks ^ mask));
        }
        return mem[bitmasks];
    }
}