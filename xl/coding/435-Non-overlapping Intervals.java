// DP or Greedy O(N log N)
class Solution {
    private int[] mem;
    private int len;

    public int eraseOverlapIntervals(int[][] intervals) {
        len = intervals.length;
        mem = new int[len];
        Arrays.fill(mem, -1);
        Arrays.sort(intervals, (a, b) -> {
            if (a[0] == b[0]) {
                return a[1] - b[1];
            }
            return a[0] - b[0];
        });
        return getMinRm(intervals, 0);
    }

    private int getMinRm(int[][] intervals, int cur) {
        // base
        if (cur >= len - 1) {
            return 0;
        }
        if (mem[cur] != -1) {
            return mem[cur];
        }
        int res = 1 + getMinRm(intervals, cur + 1);
        int next = findNext(intervals, cur);
        res = Math.min(res, next - cur - 1 + getMinRm(intervals, next));
        
        mem[cur] = res;
        return res;
    }
    private int findNext(int[][] intervals, int cur) {
        int left = cur + 1, right = len - 1;
        int mid = 0;
        int curE = intervals[cur][1];
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (intervals[mid][0] >= curE) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}