// 用stop当node
class Solution {
    public int numBusesToDestination(int[][] routes, int source, int target) {
        HashMap<Integer, List<Integer>> stopBus = new HashMap<Integer, List<Integer>>();
        for (int i = 0; i < routes.length; i++) {
            for (int stop : routes[i]) {
                stopBus.computeIfAbsent(stop, k -> new ArrayList<Integer>()).add(i);
            }
        }
        Deque<Integer> queue = new ArrayDeque<Integer>();
        HashSet<Integer> visited = new HashSet<Integer>();
        HashSet<Integer> visitedBus = new HashSet<Integer>();
        if (source == target) {
            return 0;
        }
        if (!stopBus.containsKey(source) || !stopBus.containsKey(target)) {
            return -1;
        }
        queue.offer(source);
        visited.add(source);
        int count = 0;
        while (!queue.isEmpty()) {
            count++;
            Deque<Integer> newQ = new ArrayDeque<Integer>();
            while (!queue.isEmpty()) {
                int stop = queue.poll();
                List<Integer> buses = stopBus.get(stop);
                for (int bus : buses) {
                    if (visitedBus.contains(bus)) {
                        continue;
                    }
                    visitedBus.add(bus);
                    for (int s : routes[bus]) {
                        if (s == target) {
                            return count;
                        }
                        if (visited.contains(s)) {
                            continue;
                        }
                        visited.add(s);
                        newQ.offer(s);
                    }
                }
            }
            queue = newQ;
        }

        return -1;
    }
}

////////////////////////////////////
// 也可以用bus当node