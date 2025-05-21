// O(M * 3 ^ L)
// M - the number of board cells
// L - max word length
class Solution {
    private int[][] dirs = new int[][] {
        { 1, 0},
        {-1, 0},
        { 0,  1},
        { 0, -1}
    };

    private class TrieNode {
        Map<Character, TrieNode> children;
        String word;
        public TrieNode() {
            children = new HashMap<>();
            word = null;
        }
    }
    public List<String> findWords(char[][] board, String[] words) {
        // rows > 0, cols > 0
        int rows = board.length, cols = board[0].length;
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                cur = cur.children.computeIfAbsent(c, k -> new TrieNode());
            }
            cur.word = word;
        }

        List<String> res = new ArrayList<String>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (root.children.containsKey(board[i][j])) {
                    dfs(board, i, j, root, res);
                }
            }
        }
        return res;
    }

    private void dfs(char[][] board, int r, int c, TrieNode parent, List<String> res) {
        int rows = board.length, cols = board[0].length;
        char letter = board[r][c];
        TrieNode cur = parent.children.get(letter);
        if (cur.word != null) {
            res.add(cur.word);
            cur.word = null;
        }
        if (cur.children.isEmpty()) {
            parent.children.remove(letter);
            return;
        }
        board[r][c] = '0';
        for (int[] dir : dirs) {
            int newr = r + dir[0];
            int newc = c + dir[1];
            if (newr < 0 || newr >= rows || newc < 0 || newc >= cols 
                || !cur.children.containsKey(board[newr][newc]) || board[newr][newc] == '0' ) {
                continue;
            }
            dfs(board, newr, newc, cur, res);
        }
        board[r][c] = letter;
        if (cur.children.isEmpty()) {
            parent.children.remove(letter);
        }
    }
}