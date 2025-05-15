class Solution {
    public List<String> findItinerary(List<List<String>> tickets) {
        HashMap<String, PriorityQueue<String>> edges = new HashMap<String, PriorityQueue<String>>();
        for (List<String> ticket : tickets) {
            edges.computeIfAbsent(ticket.get(0), k -> new PriorityQueue<String>()).offer(ticket.get(1));
        }
        List<String> res = new ArrayList<String>();
        dfs("JFK", edges, res);
        return res;
    }

    private void dfs(String from, HashMap<String, PriorityQueue<String>> edges, List<String> res) {
        if (!edges.containsKey(from)) {
            res.addFirst(from);
            return;
        }
        PriorityQueue<String> dests = edges.get(from);
        while (!dests.isEmpty()) {
            String dest = dests.poll();
            dfs(dest, edges, res);
        }
        res.addFirst(from);
    }
}