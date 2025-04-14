class Solution {
    public int minSteps(String s, String t) {
        int[] scount = new int[26];
        int[] tcount = new int[26];
        int len = s.length();
        for (int i = 0; i < len; i ++) {
            scount[s.charAt(i) - 'a'] ++;
            tcount[t.charAt(i) - 'a'] ++;
        }
        int result = 0;
        for (int i = 0; i < 26; i ++) {
            result += Math.max(scount[i] - tcount[i], 0);
        }
        return result;
    }
}