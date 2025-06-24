// bfs
// O(N)
class Solution {
    public String pushDominoes(String dominoes) {
        int len = dominoes.length();
        char[] darr = dominoes.toCharArray();
        HashSet<Integer> visited = new HashSet<>();
        HashMap<Integer, Integer> dirs = new HashMap<Integer, Integer>();
        Queue<Integer> q = new ArrayDeque<Integer>();
        for (int i = 0; i < darr.length; i++) {
            if (darr[i] != '.') {
                q.offer(i);
                visited.add(i);
            }
        }
        Queue<Integer> newq = new ArrayDeque<Integer>();
        while (!q.isEmpty()) {
            int cur = q.poll();
            if (darr[cur] == 'R' && cur + 1 < len && !visited.contains(cur + 1)) {
                dirs.put(cur + 1, dirs.getOrDefault(cur + 1, 0) + 1);
            } else if (darr[cur] == 'L' && cur - 1 >= 0 && !visited.contains(cur - 1)) {
                dirs.put(cur - 1, dirs.getOrDefault(cur - 1, 0) - 1);
            }
            if (q.isEmpty()) {
                newq = new ArrayDeque<>();
                for (Map.Entry<Integer, Integer> e : dirs.entrySet()) {
                    visited.add(e.getKey());
                    if (e.getValue() == 0) {
                        continue;
                    } else if (e.getValue() == 1) {
                        darr[e.getKey()] = 'R';
                        newq.offer(e.getKey());
                    } else {
                        darr[e.getKey()] = 'L';
                        newq.offer(e.getKey());
                    }
                }
                q = newq;
                dirs.clear();
            }
        }
        return new String(darr);
    }
}

// 也可以DP O(N)，从左到右计数，然后从右往左，然后看差值