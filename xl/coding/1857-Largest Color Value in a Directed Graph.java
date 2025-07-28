class Solution {
    public int largestPathValue(String colors, int[][] edges) {
        int n = colors.length();
        int[][] mem = new int[n][26];
        int[] inD = new int[n];
        int[] outD = new int[n];
        HashMap<Integer, List<Integer>> fromMap = new HashMap<>();
        for (int[] e : edges) {
            inD[e[1]]++;
            outD[e[0]]++;
            fromMap.computeIfAbsent(e[1], k -> new ArrayList<>()).add(e[0]);
        }
        // topo sort
        Deque<Integer> q = new ArrayDeque<>();
        int count = 0;
        for (int i = 0; i < outD.length; i++) {
            if (outD[i] == 0) {
                q.offer(i);
                count++;
                mem[i][colors.charAt(i) - 'a']++;
            }
        }
        while (!q.isEmpty()) {
            int cur = q.poll();
            for (int nei : fromMap.getOrDefault(cur, new ArrayList<>())) {
                outD[nei]--;
                for (int i = 0; i < mem[nei].length; i++) {
                // 26
                    mem[nei][i] = Math.max(mem[nei][i], mem[cur][i]);
                }
                if (outD[nei] == 0) {
                    q.offer(nei);
                    count++;
                    mem[nei][colors.charAt(nei) - 'a']++;
                }
            }
        }
        if (count != n) {
            return -1;
        }
        int res = 0;
        for (int i = 0; i < inD.length; i++) {
            if (inD[i] == 0) {
                for (int color : mem[i]) {
                    res = Math.max(res, color);
                }
            }
        }
        return res;
    }
}