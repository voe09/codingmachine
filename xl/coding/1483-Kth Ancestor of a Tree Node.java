class TreeAncestor {
    private int[][] ancestor;
    private int height;

    public TreeAncestor(int n, int[] parent) {
        height = 0;
        int cur = n;
        while (cur > 0) {
            cur >>= 1;
            height++;
        }
        ancestor = new int[height][n];
        ancestor[0] = parent;
        buildAncestorMap(n);
    }
    
    public int getKthAncestor(int node, int k) {
        int shift = 0;
        while (shift < height && node >= 0) {
            if ((k & (1 << shift)) > 0) {
                node = ancestor[shift][node];
            }
            shift++;
        }
        return node;
    }

    private void buildAncestorMap(int n) {
        for (int power = 1; power < height; power++) {
            for (int i = 0; i < n; i++) {
                int halfAnc = ancestor[power - 1][i];
                ancestor[power][i] = halfAnc == -1 ? -1 : ancestor[power - 1][halfAnc];
            }
        }
    }
}

/**
 * Your TreeAncestor object will be instantiated and called as such:
 * TreeAncestor obj = new TreeAncestor(n, parent);
 * int param_1 = obj.getKthAncestor(node,k);
 */