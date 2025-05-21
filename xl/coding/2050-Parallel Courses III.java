// O(N + E)
class Solution {
    public int minimumTime(int n, int[][] relations, int[] time) {
        int[] indegree = new int[n];
        int[] preTime = new int[n];
        HashMap<Integer, List<Integer>> edges = new HashMap<>();
        int res = 0;
        for (int[] edge : relations) {
            int prev = edge[0] - 1;
            int next = edge[1] - 1;
            edges.computeIfAbsent(prev, k -> new ArrayList<Integer>()).add(next);
            indegree[next]++;
        }
        Queue<Integer> q = new ArrayDeque<Integer>();
        for (int i = 0; i < indegree.length; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
            }
        }

        while (!q.isEmpty()) {
            int cur = q.poll();
            res = Math.max(res, time[cur] + preTime[cur]);
            if (edges.containsKey(cur)) {
                List<Integer> neis = edges.get(cur);
                for (int nei : neis) {
                    indegree[nei]--;
                    if (indegree[nei] == 0) {
                        q.offer(nei);
                    }
                    preTime[nei] = Math.max(preTime[nei], preTime[cur] + time[cur]);
                }
            }
        }
        return res;
    }
}