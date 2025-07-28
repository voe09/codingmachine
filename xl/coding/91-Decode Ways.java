class Solution {
    public int numDecodings(String s) {
        int len = s.length();
        if (len == 0) {
            return 0;
        }
        int[] dp = new int[len + 1];
        dp[0] = 1;
        if (s.charAt(0) != '0') {
            dp[0 + 1] = 1;
        }
        
        for (int i = 1; i < len; i++) {
            char c = s.charAt(i);
            // from s[i - 1]
            if (c != '0') {
                dp[i + 1] += dp[i];
            }
            // from s[i - 2]
            if ( s.charAt(i - 1) == '1' || (s.charAt(i - 1) == '2' && s.charAt(i) <= '6') ) {
                dp[i + 1] += dp[i - 1];
            }
        }
        return dp[len];
    }
}