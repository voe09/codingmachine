class Solution {
    public boolean buddyStrings(String s, String goal) {
        if (s.length() != goal.length()) return false;
        List<Integer> diff = new ArrayList<>();
        int n = s.length();
        char diffl = ' ';
        char diffr = ' ';
        for (int i = 0; i < n; i ++){
            if (s.charAt(i) != goal.charAt(i)) {
                if (diffl != ' ' && (diffl != goal.charAt(i)|| diffr != s.charAt(i))) {
                    return false;
                } else if (diffl == ' ') {
                    diffl = s.charAt(i);
                    diffr = goal.charAt(i);
                }
                diff.add(i);
                if (diff.size() > 2) return false;
            }
        }
        if (diff.size() == 0) {
            Set<Character> set = new HashSet<>();
            for (int i = 0; i < s.length(); i ++) {
                if (set.contains(s.charAt(i))) {
                    return true;
                }
                set.add(s.charAt(i));
            }
            return false;
        }
        return diff.size() == 2;
    }
}