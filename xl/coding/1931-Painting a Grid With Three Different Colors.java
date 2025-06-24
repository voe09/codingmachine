class Solution {
    public int colorTheGrid(int m, int n) {
        final int MOD = (int)Math.pow(10, 9) + 7;
        // m << n
        int singleRow = (int)Math.pow(3, m);
        // decimal to ternary
        HashMap<Integer, List<Integer>> decToTer = new HashMap<Integer, List<Integer>>();
        for (int i = 0; i < singleRow; i++) {
            int cur = i;
            List<Integer> ternary = new ArrayList<>();
            boolean valid = true;
            while (ternary.size() < m) {
                if (ternary.size() > 0 && ternary.get(ternary.size() - 1) == cur % 3) {
                    valid = false;
                    break;
                }
                ternary.add(cur % 3);
                cur /= 3;
            }
            if (valid) {
               decToTer.put(i, ternary);
            }
        }

        // pre-find all valid adjacent combination for each ternary
        HashMap<Integer, List<Integer>> adjs = new HashMap<>();
        for (int cur : decToTer.keySet()) {
            List<Integer> curTer = decToTer.get(cur);
            for (int nei : decToTer.keySet()) {
                List<Integer> neiTer = decToTer.get(nei);
                boolean valid = true;
                for (int i = 0; i < curTer.size(); i++) {
                    if (curTer.get(i) == neiTer.get(i)) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    adjs.computeIfAbsent(cur, k -> new ArrayList<Integer>()).add(nei);
                }
            }
        }

        // dp
        long[][] dp = new long[n][singleRow];
        for (int cur : decToTer.keySet()) {
            dp[0][cur] = 1;
        }
        for (int i = 1; i < dp.length; i++) {
            for (int cur : decToTer.keySet()) {
                for (int adj : adjs.get(cur)) {
                    dp[i][cur] += dp[i - 1][adj];
                    dp[i][cur] %= MOD;
                }
            }
        }
        long res = 0;
        for (long val : dp[n - 1]) {
            res += val;
            res %= MOD;
        }
        return (int)res;
    }
}