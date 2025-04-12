

class Solution {
    class Task {
        int start;
        int end;
        int profit;
        public Task(int s, int e, int p) {
            start = s;
            end = e;
            profit = p;
        }
    }
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        List<Task> taskList = new ArrayList<>();
        for (int i = 0; i < n; i ++) {
            taskList.add(new Task(startTime[i], endTime[i], profit[i]));
        }
        Collections.sort(taskList, (a, b) -> {
            return a.end - b.end;
        });
        TreeMap<Integer, Integer> dp = new TreeMap<Integer, Integer>();
        dp.put(0, 0);
        for (int i = 0; i < n; i ++) {
            Task t = taskList.get(i);
            int cur = 0;
            cur = Math.max(dp.floorEntry(t.start).getValue() + t.profit, dp.lastEntry().getValue());
            dp.put(t.end, cur);
        }
        return dp.lastEntry().getValue();
    }
}