// O(char_len + m * m)
// char_len - total chars count in products
// m - searchWord length
class Solution {
    private class TrieNode {
        TrieNode[] children;
        String endWord;
        List<String> words;

        public TrieNode() {
            children = new TrieNode[26];
            endWord = null;
            words = new ArrayList<String>();
        }
    }
    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        Arrays.sort(products);
        TrieNode root = new TrieNode();
        TrieNode cur = root;
        for (String prod : products) {
            cur = root;
            for (char c : prod.toCharArray()) {
                cur.words.add(prod);
                if (cur.children[c - 'a'] == null) {
                    cur.children[c - 'a'] = new TrieNode();
                }
                cur = cur.children[c - 'a'];
            }
            cur.words.add(prod);
            cur.endWord = prod;
        }
        List<List<String>> res = new ArrayList<>();
        cur = root;
        for (char c : searchWord.toCharArray()) {
            List<String> sug = new ArrayList<>();
            if (cur != null && cur.children[c - 'a'] != null) {
                cur = cur.children[c - 'a'];
                for (int i = 0; i < Math.min(3, cur.words.size()); i++) {
                    sug.add(cur.words.get(i));
                }
            } else {
                cur = null;
            }
            res.add(sug);
        }
        return res;
    }
}