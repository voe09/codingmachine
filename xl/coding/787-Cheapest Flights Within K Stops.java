class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        int[] stops = new int[n];
        Arrays.fill(stops, Integer.MAX_VALUE);
        HashMap<Integer, List<int[]>> neis = new HashMap<>();
        for (int[] f : flights) {
            neis.computeIfAbsent(f[0], key -> new ArrayList<int[]>()).add(new int[] { f[1], f[2] });
        }
        // { city, price, stop }
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>( (a, b) -> a[1] - b[1]);
        pq.offer(new int[] { src, 0, 0 });
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int city = cur[0];
            if (city == dst) {
                return cur[1];
            }
            if (!neis.containsKey(city) || stops[city] <= cur[2]) {
                continue;
            }
            stops[city] = cur[2];
            for (int[] next : neis.get(city)) {
                if (cur[2] < k || next[0] == dst) {
                    pq.offer(new int[] { next[0], cur[1] + next[1], cur[2] + 1 });
                }
            }
        }
        return -1;
    }
}