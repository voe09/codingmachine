class Solution {
    public boolean areAlmostEqual(String s1, String s2) {
        int left = 0;
        int right = 0;
        if (s1.length() != s2.length()) {
            return false;
        }
        int len = s1.length();
        int diff = 0;
        char diffl = ' ';
        char diffr = ' ';
        while (left < len) {
            if (s1.charAt(left) != s2.charAt(right)) {
                if (diffl != ' ' && (diffl != s2.charAt(right) || diffr != s1.charAt(left))) {
                    return false;
                } else if (diffl == ' ') {
                    diffl = s1.charAt(left);
                    diffr = s2.charAt(right);
                }
                diff ++;
            }
            if (diff > 2) return false;
            left ++;
            right ++;
        }
        return diff == 2 || diff == 0;
    }
}