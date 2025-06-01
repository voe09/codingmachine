// O(MN log MAX_DIFF)
// MAX_DIFF = 10 ^ 6 - 0
// Binary Search + dfs
class Solution {
    private int[][] dirs = new int[][] {
        { 1, 0},
        {-1, 0},
        { 0,  1},
        { 0, -1}
    };
    private int rows;
    private int cols;
    public int minimumEffortPath(int[][] heights) {
        // rows > 0, cols > 0
        rows = heights.length; cols = heights[0].length;
        int start = 0, end = (int)Math.pow(10, 6);
        int mid = 0;
        while (start < end) {
            mid = (end - start) / 2 + start;
            boolean[][] visited = new boolean[rows][cols];
            if (canReachDest(heights, visited, mid, 0, 0)) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return end;
    }

    private boolean canReachDest(int[][]heights, boolean[][] visited, int targetDiff, int r, int c) {
        if (r == rows - 1 && c == cols - 1) {
            return true;
        }
        visited[r][c] = true;
        for (int[] dir : dirs) {
            int newr = r + dir[0];
            int newc = c + dir[1];
            if (newr < 0 || newr >= rows || newc < 0 || newc >= cols || visited[newr][newc]
                || Math.abs(heights[r][c] - heights[newr][newc]) > targetDiff) {
                continue;
            }
            if (canReachDest(heights, visited, targetDiff, newr, newc)) {
                return true;
            }
        }
        return false;
    }
}

// O(MN log MN)
// dijkstra
class Solution1 {
    public int minimumEffortPath(int[][] heights) {
        int[][] dirs = new int[][] {
            { 1, 0},
            {-1, 0},
            { 0,  1},
            { 0, -1}
        };
        // rows > 0, cols > 0
        int rows = heights.length, cols = heights[0].length;
        // a[0] - row
        // a[1] - col
        // a[2] - diff
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        int res = 0;
        pq.offer(new int[] { 0, 0, 0 });
        int[][] diffMatrix = new int[rows][cols];
        for (int[] row : diffMatrix) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int r = cur[0], c = cur[1];
            if (diffMatrix[r][c] < cur[2]) {
                continue;
            }
            diffMatrix[r][c] = -1;
            if (r == rows - 1 && c == cols - 1) {
                res = cur[2];
                break;
            }
            for (int[] dir : dirs) {
                int newr = r + dir[0];
                int newc = c + dir[1];
                if (newr < 0 || newr >= rows || newc < 0 || newc >= cols || diffMatrix[newr][newc] == -1) {
                    continue;
                }
                int diff = Math.max(cur[2], Math.abs(heights[r][c] - heights[newr][newc]));
                diffMatrix[newr][newc] = Math.min(diff, diffMatrix[newr][newc]);
                pq.offer(new int[] { newr, newc, diff });
            }
        }
        return res;
    }
}