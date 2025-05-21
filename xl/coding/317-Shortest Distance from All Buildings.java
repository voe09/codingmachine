// O((MN)^2)
class Solution {
    public int shortestDistance(int[][] grid) {
        // rows > 0, cols > 0
        int rows = grid.length, cols = grid[0].length;
        int[][] dist = new int[rows][cols];
        int[][] visited = new int[rows][cols];
        int bldCount = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    Queue<int[]> q = new ArrayDeque<int[]>();
                    q.offer(new int[] { i, j });
                    bfs(grid, q, dist, visited, bldCount);
                    bldCount++;
                }
            }
        }

        int res = -1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (visited[i][j] != bldCount) {
                    continue;
                }
                if (res == -1) {
                    res = dist[i][j];
                } else {
                    res = Math.min(res, dist[i][j]);
                }
            }
        }

        return res;
    }

    private void bfs(int[][] grid, Queue<int[]> q, int[][] dist, int[][] visited, int bldCount) {
        int rows = grid.length, cols = grid[0].length;
        int[][] dirs = new int[][] {
            { 1, 0},
            {-1, 0},
            { 0,  1},
            { 0, -1}
        };
        Queue<int[]> next = new ArrayDeque<int[]>();
        int distance = 1;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            for (int[] dir : dirs) {
                int r = cur[0] + dir[0];
                int c = cur[1] + dir[1];
                if (r < 0 || r >= rows || c < 0 || c >= cols 
                    || grid[r][c] != 0 || visited[r][c] != bldCount) {
                    continue;
                }
                next.offer(new int[] { r, c });
                visited[r][c] = bldCount + 1;
                dist[r][c] += distance;
            }
            if (q.isEmpty()) {
                q = next;
                next = new ArrayDeque<>();
                distance++;
            }
        }
    }
}