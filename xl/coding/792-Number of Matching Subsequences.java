class Solution {
    public int numMatchingSubseq(String s, String[] words) {
        int len = s.length();
        HashMap<Character, ArrayList<int[]>> groups = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            char start = words[i].charAt(0);
            groups.computeIfAbsent(start, k -> new ArrayList<>()).add(new int[] { i, 0 });
        }
        int res = 0;
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (!groups.containsKey(c)) {
                continue;
            }
            List<int[]> cur = groups.get(c);
            groups.put(c, new ArrayList<>());
            for (int[] node : cur) {
                String word = words[node[0]];
                int idx = node[1] + 1;
                if (idx == word.length()) {
                    res++;
                } else {
                    groups.computeIfAbsent(word.charAt(idx), k -> new ArrayList<>()).add(new int[] { node[0], idx });
                }
            }
        }
        return res;
    }
}