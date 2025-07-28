class Solution {
    public int calculateMinimumHP(int[][] dungeon) {
        // rows, cols > 0
        int rows = dungeon.length, cols = dungeon[0].length;
        int[][] dp = new int[rows][cols];
        for (int[] row : dp) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        dp[rows - 1][cols - 1] = Math.max(-dungeon[rows - 1][cols - 1] + 1, 1);
        Queue<int[]> q = new ArrayDeque<int[]>();
        q.offer(new int[] { rows - 1, cols - 1 });
        boolean[][] inQ = new boolean[rows][cols];
        // bellman ford
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int r = cur[0], c = cur[1];
            inQ[r][c] = false;
            if (r - 1 >= 0) {
                int health = Math.max(1, dp[r][c] - dungeon[r - 1][c]);
                if (dp[r - 1][c] > health) {
                    dp[r - 1][c] = health;
                    if (!inQ[r - 1][c]) {
                        q.offer(new int[] { r - 1, c });
                    }
                }
            }
            if (c - 1 >= 0) {
                int health = Math.max(1, dp[r][c] - dungeon[r][c - 1]);
                if (dp[r][c - 1] > health) {
                    dp[r][c - 1] = health;
                    if (!inQ[r][c - 1]) {
                        q.offer(new int[] { r, c - 1 });
                    }
                }
            }
        }
        return dp[0][0];
    }
}