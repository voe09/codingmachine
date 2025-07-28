class Solution {
    private int[] mem;
    public int numTrees(int n) {
        mem = new int[n + 1];
        mem[1] = 1;
        return dfs(1, n);
    }

    private int dfs(int leftB, int rightB) {
        int delta = rightB - leftB + 1;
        if (mem[delta] != 0) {
            return mem[delta];
        }
        int sum = dfs(leftB + 1, rightB) + dfs(leftB, rightB - 1);
        for (int i = leftB + 1; i < rightB; i++) {
            int left = dfs(leftB, i - 1);
            int right = dfs(i + 1, rightB);
            sum += left * right;
        }
        mem[delta] = sum;
        return mem[delta];
    }
}