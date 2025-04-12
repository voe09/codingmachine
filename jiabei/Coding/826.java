class Solution {
    class Task{
        int difficulty;
        int profit;
        public Task(int d, int p) {
            difficulty = d;
            profit = p;
        }
    }
    public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
        int n = profit.length;
        int m = worker.length;
        List<Task> taskList = new ArrayList<>();
        List<Integer> workerList = new ArrayList<>();
        for (int i = 0; i < n; i ++) {
            taskList.add(new Task(difficulty[i], profit[i]));
        }
        for (int i = 0; i < m; i ++) {
            workerList.add(worker[i]);
        }
        Collections.sort(taskList, (a, b) -> {
            return b.profit - a.profit;
        });
        int result = 0;
        Collections.sort(workerList, (a, b) -> {
            return b - a;
        });
        int idx = 0;
        for (int w : workerList){
            while (idx < n && w < taskList.get(idx).difficulty) {
                idx ++;
            }
            if (idx == n) break;
            result += taskList.get(idx).profit;
        }
        return result;
    }
}