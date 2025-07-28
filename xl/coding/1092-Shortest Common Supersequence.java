class Solution {
    public String shortestCommonSupersequence(String str1, String str2) {
        int len1 = str1.length(), len2 = str2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        for (int j = 0; j < dp[0].length; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i < dp.length; i++) {
            dp[i][0] = i;
            for (int j = 1; j < dp[i].length; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        int r = len1, c = len2;
        StringBuilder sb = new StringBuilder();
        while (r > 0 && c > 0) {
            if (str1.charAt(r - 1) == str2.charAt(c - 1)) {
                sb.append(str1.charAt(r - 1));
                r--;
                c--;
            } else if (dp[r][c - 1] < dp[r - 1][c]) {
                sb.append(str2.charAt(c - 1));
                c--;
            } else {
                sb.append(str1.charAt(r - 1));
                r--;
            }
        }
        while (r > 0) {
            sb.append(str1.charAt(r - 1));
            r--;
        }
        while (c > 0) {
            sb.append(str2.charAt(c - 1));
            c--;
        }
        return sb.reverse().toString();
    }
}