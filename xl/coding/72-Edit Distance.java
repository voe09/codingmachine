class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] mem = new int[m + 1][n + 1];
        calDist(word1, word2, 0, 0, mem);
        return mem[0][0];
    }

    private int calDist(String word1, String word2, int a, int b, int[][] mem) {
        if (a == word1.length()) {
            mem[a][b] = word2.length() - b;
            return mem[a][b];
        }
        if (b == word2.length()) {
            mem[a][b] = word1.length() - a;
            return mem[a][b];
        }
        if (mem[a][b] > 0) {
            return mem[a][b];
        }
        if (word1.charAt(a) == word2.charAt(b)) {
            mem[a][b] = calDist(word1, word2, a + 1, b + 1, mem);
        } else {
            int add = calDist(word1, word2, a, b + 1, mem) + 1;
            int delete = calDist(word1, word2, a + 1, b, mem) + 1;
            int replace = calDist(word1, word2, a + 1, b + 1, mem) + 1;
            mem[a][b] = Math.min(add, Math.min(delete, replace));
        }
        return mem[a][b];
    }
}

class Solution1 {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i][j - 1], Math.min(dp[i - 1][j], dp[i - 1][j - 1])) + 1;
                }
            }
        }
        return dp[m][n];
    }
}