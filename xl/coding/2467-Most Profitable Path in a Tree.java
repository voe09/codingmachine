// O(N)
class Solution {
    public int mostProfitablePath(int[][] edges, int bob, int[] amount) {
        HashMap<Integer, List<Integer>> edgeMap = new HashMap<>();
        for (int[] edge : edges) {
            edgeMap.computeIfAbsent(edge[0], k -> new ArrayList<Integer>()).add(edge[1]);
            edgeMap.computeIfAbsent(edge[1], k -> new ArrayList<Integer>()).add(edge[0]);
        }
        int[] distance = new int[amount.length];
        boolean[] visited = new boolean[amount.length];
        return dfs(edgeMap, amount, bob, 0, 0, distance, visited);
    }

    private int dfs(HashMap<Integer, List<Integer>> edgeMap, int[] amount, int bob, int cur, int level, int[] distance, boolean[] visited) {
        if (cur == bob) {
            distance[cur] = 0;
        } else {
            distance[cur] = amount.length;
        }
        int maxProfit = Integer.MIN_VALUE;
        visited[cur] = true;
        List<Integer> neis = edgeMap.get(cur);
        for (int nei : neis) {
            if (visited[nei]) {
                continue;
            }
            maxProfit = Math.max(maxProfit, dfs(edgeMap, amount, bob, nei, level + 1, distance, visited));
            if (distance[nei] != amount.length) {
                distance[cur] = distance[nei] + 1;
            }
        }
        if (maxProfit == Integer.MIN_VALUE) {
            maxProfit = 0;
        }
        if (level == distance[cur]) {
            maxProfit += amount[cur] / 2;
        } else if (level < distance[cur]) {
            maxProfit += amount[cur];
        }
        return maxProfit;
    }
}