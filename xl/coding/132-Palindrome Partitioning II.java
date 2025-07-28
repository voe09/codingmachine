class Solution {
    private int[] mem;

    public int minCut(String s) {
        int len = s.length();
        // pre-compute isPalindrome
        boolean[][] isPal = new boolean[len][len];
        for (int i = len - 1; i >= 0; i--) {
            isPal[i][i] = true;
            for (int j = i + 1; j < len; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j == i + 1) {
                        isPal[i][j] = true;
                    } else {
                        isPal[i][j] = isPal[i + 1][j - 1];
                    }
                }
            }
        }
        // recursion + mem
        mem = new int[len];
        Arrays.fill(mem, -1);
        return getMinCut(s, 0, isPal) - 1;
    }

    private int getMinCut(String s, int start, boolean[][] isPal) {
        if (start == s.length()) {
            return 0;
        }
        if (mem[start] != -1) {
            return mem[start];
        }
        int res = s.length();
        for (int i = start; i < s.length(); i++) {
            if (isPal[start][i]) {
                res = Math.min(res, 1 + getMinCut(s, i + 1, isPal));
            }
        }
        mem[start] = res;
        return res;
    }
}