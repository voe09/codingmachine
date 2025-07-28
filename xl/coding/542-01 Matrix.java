class Solution {
    public int[][] updateMatrix(int[][] mat) {
        final int[][] dirs = new int[][] {
            { 1, 0},
            {-1, 0},
            { 0,  1},
            { 0, -1}
        };
        // rows, cols > 0
        int rows = mat.length, cols = mat[0].length;
        int[][] res = new int[rows][cols];
        for (int[] row : res) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        Queue<int[]> q = new ArrayDeque<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mat[i][j] == 0) {
                    q.offer(new int[] { i, j });
                    res[i][j] = 0;
                }
            }
        }
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            for (int[] dir : dirs) {
                int newR = cur[0] + dir[0];
                int newC = cur[1] + dir[1];
                if (newR < 0 || newR >= rows || newC < 0 || newC >= cols 
                    || mat[newR][newC] == 0 || res[newR][newC] <= res[cur[0]][cur[1]] + 1) {
                    continue;
                }
                res[newR][newC] = res[cur[0]][cur[1]] + 1;
                q.offer(new int[] { newR, newC });
            }
        }
        return res;
    }
}