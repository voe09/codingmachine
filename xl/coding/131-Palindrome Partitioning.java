class Solution {
    HashMap<Integer, List<List<String>>> mem;
    public List<List<String>> partition(String s) {
        int len = s.length();
        mem = new HashMap<>();
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

        return getPalList(s, 0, isPal);
    }

    private List<List<String>> getPalList(String s, int start, boolean[][] isPal) {
        List<List<String>> res = new ArrayList<>();
        if (start == s.length()) {
            res.add(new ArrayList<>());
            return res;
        }
        if (mem.containsKey(start)) {
            return mem.get(start);
        }
        for (int i = start; i < s.length(); i++) {
            if (isPal[start][i]) {
                String curStr = s.substring(start, i + 1);
                List<List<String>> next = getPalList(s, i + 1, isPal);
                for (List<String> val : next) {
                    List<String> cur = new ArrayList<>();
                    cur.add(curStr);
                    cur.addAll(val);
                    res.add(cur);
                }
            }
        }
        mem.put(start, res);
        return res;
    }
}
