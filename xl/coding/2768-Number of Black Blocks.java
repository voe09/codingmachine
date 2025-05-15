class Solution {
    private int[][] dirs = new int[][] {
        { -1, -1},
        { -1,  0},
        {  0, -1},
        {  0,  0}
    };

    public long[] countBlackBlocks(int m, int n, int[][] coordinates) {
        long total = (long)(m - 1) * (n - 1);
        long[] res = new long[5];
        res[0] = total;
        HashMap<Long, Integer> count = new HashMap<Long, Integer>();
        for (int[] coor : coordinates) {
            for (int[] dir : dirs) {
                long x = coor[0] + dir[0];
                long y = coor[1] + dir[1];
                if (x < 0 || x >= m - 1 || y < 0 || y >= n - 1) {
                    continue;
                }
                int prev = count.getOrDefault(x * n + y, 0);
                int cur = prev + 1;
                count.put(x * n + y, cur);
                res[prev]--;
                res[cur]++;
            }
        }
        return res;
    }
}