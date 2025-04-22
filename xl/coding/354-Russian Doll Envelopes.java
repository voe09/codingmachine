// 300-Longest Increasing Subsequence 老年加强版 
// O(N log N): 相同宽度的，按高度从大到小排序，因为宽相同的放不进去，只要高度开始递增，就代表宽也开始递增

class Solution {
    public int maxEnvelopes(int[][] envelopes) {
        int len = envelopes.length;
        Arrays.sort(envelopes, (a, b) -> {
            if (a[0] != b[0]) {
                return a[0] - b[0];
            } else {
                return b[1] - a[1];
            }
        });

        int[] envBig = new int[len];
        envBig[0] = envelopes[0][1];
        int end = 0;
        // int res = 1;
        for (int i = 1; i < len; i++) {
            end = findPos(envBig, 0, end, envelopes[i][1]);
            // res = Math.max(res, end + 1);
        }

        return end + 1;
    }

    private int findPos(int[] envBig, int start, int end, int h) {
        int oriEnd = end;
        while (start <= end) {
            int mid = (end - start) / 2 + start;
            if (envBig[mid] < h) {
                start = mid + 1;
            } else if (envBig[mid] > h) {
                end = mid - 1;
            } else {
                start = mid;
                break;
            }
        }
        envBig[start] = h;
        return Math.max(start, oriEnd);
    }
}