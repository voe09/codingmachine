// O(N log N)
class Solution {
    public int[] getOrder(int[][] tasks) {
        // len > 0
        int len = tasks.length;
        int[][] tasksWithId = new int[len][3];
        int[] res = new int[len];
        for (int i = 0; i < len; i++) {
            tasksWithId[i] = new int[] { i, tasks[i][0], tasks[i][1] };
        }
        Arrays.sort(tasksWithId, (a, b) -> a[1] - b[1]);
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[2] - b[2] == 0) {
                return a[0] - b[0];
            } else {
                return a[2] - b[2];
            }
        });
        int curTime = tasksWithId[0][1];
        int next = 0;
        int resIdx = 0;
        while (next < len && tasksWithId[next][1] <= curTime) {
            pq.offer(tasksWithId[next]);
            next++;
        }
        while (!pq.isEmpty()) {
            int[] curTask = pq.poll();
            res[resIdx] = curTask[0];
            curTime += curTask[2];
            if (pq.isEmpty() && next < len && tasksWithId[next][1] > curTime) {
                curTime = tasksWithId[next][1];
            }
            while (next < len && tasksWithId[next][1] <= curTime) {
                pq.offer(tasksWithId[next]);
                next++;
            }
            resIdx++;
        }
        return res;
    }
}