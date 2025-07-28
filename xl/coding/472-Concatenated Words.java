// O(N M^2)
// N - words.length
// M - words[i] avg length
class Solution {
    private class TrieNode {
        TrieNode[] children;
        String word;

        public TrieNode() {
            children = new TrieNode[26];
            word = null;
        }
    }

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        int len = words.length;
        TrieNode root = new TrieNode();
        // build trie tree
        for (String word : words) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                if (cur.children[c - 'a'] == null) {
                    cur.children[c - 'a'] = new TrieNode();
                }
                cur = cur.children[c - 'a'];
            }
            cur.word = word;
        }
        // dp
        List<String> res = new ArrayList<>();
        for (String word : words) {
            int wlen = word.length();
            boolean[] dp = new boolean[wlen + 1];
            dp[wlen] = true;
            for (int i = wlen - 1; i >= 0; i--) {
                TrieNode cur = root;
                for (int j = i; j < wlen; j++) {
                    char c = word.charAt(j);
                    cur = cur.children[c - 'a'];
                    if (cur == null) {
                        break;
                    }
                    if (cur.word != null && dp[j + 1] && cur.word.length() != word.length()) {
                    // if (cur.word != null && dp[j + 1] && !(i == 0 && j == wlen - 1)) {
                        dp[i] = true;
                        break;
                    }
                }
            }
            if (dp[0]) {
                res.add(word);
            }
        }
        return res;
    }
}