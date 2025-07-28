class Solution {
    private HashMap<Integer, Integer> mem;

    public int getKth(int lo, int hi, int k) {
        mem = new HashMap<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[0] == b[0]) {
                return b[1] - a[1];
            } else {
                return b[0] - a[0];
            }
        });
        for (int i = lo; i <= hi; i++) {
            int power = getPower(i);
            pq.offer(new int[] { power, i });
            if (pq.size() > k) {
                pq.poll();
            }
        }
        return pq.peek()[1];
    }

    private int getPower(int num) {
        if (num == 1) {
            return 0;
        }
        if (mem.containsKey(num)) {
            return mem.get(num);
        }
        if (num % 2 == 0) {
            mem.put(num, getPower(num / 2) + 1);
            return mem.get(num);
        } else {
            mem.put(num, getPower(num * 3 + 1) + 1);
            return mem.get(num);
        }
    }
}