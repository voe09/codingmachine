class Solution {
    private int[][] dirs = new int[][] {
        { 1, 0},
        {-1, 0},
        { 0,  1},
        { 0, -1}
    };
    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        int rows = maze.length;
        int cols = maze[0].length;
        // rows > 0, cols > 0, start != dest, start == 0, dest == 0
        int[][] dist = new int[rows][cols];
        for (int[] row : dist) {
            Arrays.fill(row, -1);
        }
        dist[start[0]][start[1]] = 0;
        dijkstra(maze, start, dist);
        return dist[destination[0]][destination[1]];
    }

    private void dijkstra(int[][] maze, int[] start, int[][] dist) {
        int rows = maze.length;
        int cols = maze[0].length;
        PriorityQueue<int[]> pq = new PriorityQueue<int[]> ((a, b) -> a[2] - b[2]);
        pq.offer(new int[] { start[0], start[1], 0 });
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int x = cur[0];
            int y = cur[1];
            if (dist[x][y] != -1 && dist[x][y] < cur[2]) {
                continue;
            }
            for (int[] dir : dirs) {
                int distance = -1;
                x = cur[0];
                y = cur[1];
                while (x >= 0 && x < rows && y >= 0 && y < cols && maze[x][y] == 0) {
                    x += dir[0];
                    y += dir[1];
                    distance++;
                }
                x -= dir[0];
                y -= dir[1];
                // distance += cur[2];
                distance += dist[cur[0]][cur[1]];

                // if (x == cur[0] && y == cur[1]) {
                //     continue;
                // }
                if (dist[x][y] == -1 || dist[x][y] > distance) {
                    dist[x][y] = distance;
                    pq.offer(new int[] { x, y, distance });
                }
            }
        }
    }
}