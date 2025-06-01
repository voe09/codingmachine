// O(N * L)
// N - folder array length
// L - longest path length
class Solution {
    private class TrieNode {
        TrieNode[] children;
        boolean isEnd;

        public TrieNode() {
            children = new TrieNode[27];
            isEnd = false;
        }
    }
    public List<String> removeSubfolders(String[] folder) {
        int len = folder.length;
        TrieNode root = new TrieNode();
        List<String> toCheck = new ArrayList<>();
        for (String name : folder) {
            TrieNode cur = root;
            boolean isSub = false;
            for (int i = 0; i < name.length(); i++) {
                char c = name.charAt(i);
                int idx = c - 'a';
                if (c == '/') {
                    idx = 26;
                }
                if (cur.children[idx] == null) {
                    cur.children[idx] = new TrieNode();
                }
                cur = cur.children[idx];
                // don't build Trie for sub folders
                if (cur.isEnd && i < name.length() - 1 && name.charAt(i + 1) == '/') {
                    isSub = true;
                    break;
                }
            }
            cur.isEnd = true;
            if (!isSub) {
                toCheck.add(name);
            }
        }
        List<String> res = new ArrayList<>();
        for (String name : toCheck) {
            TrieNode cur = root;
            boolean isSub = false;
            for (int i = 0; i < name.length(); i++) {
                char c = name.charAt(i);
                int idx = c - 'a';
                if (c == '/') {
                    idx = 26;
                }
                cur = cur.children[idx];
                if (cur.isEnd && i < name.length() - 1 && name.charAt(i + 1) == '/') {
                    isSub = true;
                    break;
                }
            }
            if (!isSub) {
                res.add(name);
            }
        }
        return res;
    }
}