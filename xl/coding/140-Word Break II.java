class Solution {
    class TrieNode {
        TrieNode[] children;
        String word;

        public TrieNode() {
            // only lowercase chars
            children = new TrieNode[26];
            word = null;
        }
    }
    public List<String> wordBreak(String s, List<String> wordDict) {
        TrieNode root = new TrieNode();
        for (String word : wordDict) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                if (cur.children[c - 'a'] == null) {
                    cur.children[c - 'a'] = new TrieNode();
                }
                cur = cur.children[c - 'a'];
            }
            cur.word = word;
        }

        Map<Integer, List<String>> mem = new HashMap<>();
        dfs(s, 0, mem, root);
        return mem.get(0);
    }

    private List<String> dfs(String s, int start, Map<Integer, List<String>> mem, TrieNode root) {
        if (mem.containsKey(start)) {
            return mem.get(start);
        }
        List<String> res = new ArrayList<String>();
        if (start == s.length()) {
            res.add("");
            return res;
        }
        TrieNode cur = root;
        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (cur.children[c - 'a'] == null) {
                break;
            }
            cur = cur.children[c - 'a'];
            if (cur.word != null) {
                List<String> next = dfs(s, i + 1, mem, root);
                for (String sentence : next) {
                    if (sentence.length() == 0) {
                        res.add(cur.word);
                    } else {
                        res.add(cur.word + " " + sentence);
                    }
                }
            }
        }
        mem.put(start, res);
        return res;
    }
}