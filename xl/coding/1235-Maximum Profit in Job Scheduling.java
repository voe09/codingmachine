// O(N log N)
class Solution {
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int len = startTime.length;
        // TreeMap<Integer, List<List<Integer>>> map = new TreeMap<>();
        HashMap<Integer, List<List<Integer>>> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            map.computeIfAbsent(endTime[i], k -> new ArrayList<List<Integer>>()).add(new ArrayList<Integer>(List.of(startTime[i], profit[i])));
        }
        // TreeMap<Integer, Integer> dp = new TreeMap<>();
        // dp.put(0, 0);

        int[][] dp = new int[map.size() + 1][2];
        int idx = 1;
        for (int end : map.keySet()) {
            dp[idx][0] = end;
            idx++;
        }
        Arrays.sort(dp, (a, b) -> a[0] - b[0]);

        int maxProfit = 0;
        for (int i = 1; i < dp.length; i++) {
            int end = dp[i][0];
            List<List<Integer>> jobs = map.get(end);
            for (List<Integer> job : jobs) {
                int prev = binarySearch(dp, i, job.get(0));
                dp[i][1] = Math.max(dp[i][1], dp[prev][1] + job.get(1));
                // dp.put(end, Math.max(dp.getOrDefault(end, 0), dp.floorEntry(job.get(0)).getValue() + job.get(1)));
            }
            // if (dp.get(end) > maxProfit) {
            //     maxProfit = dp.get(end);
            // } else {
            //     dp.put(end, maxProfit);
            // }
            if (dp[i][1] > maxProfit) {
                maxProfit = dp[i][1];
            } else {
                dp[i][1] = maxProfit;
            }
        }

        return maxProfit;
    }

    private int binarySearch(int[][] dp, int end, int target) {
        int start = 1;
        int mid = (end - start) / 2 + start;
        while (start <= end) {
            mid = (end - start) / 2 + start;
            if (dp[mid][0] == target) {
                return mid;
            }
            if (dp[mid][0] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return end;
    }
}