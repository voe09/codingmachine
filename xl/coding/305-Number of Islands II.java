// O(m * n + l)
// 可以不用 set，用 parent[]记录是不是 land，能快一点，省点空间
class Solution {
    int[][] dirs = new int[][] {
        { 1, 0},
        {-1, 0},
        {0,  1},
        {0, -1}
    };
    class UnionFind {
        int[] parent;
        int[] size;

        public UnionFind(int totalCount) {
            parent = new int[totalCount];
            size = new int[totalCount];
            for (int i = 0; i < parent.length; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        public int find(int x) {
            if (parent[x] == x) {
                return x;
            }
            parent[x] = find(parent[x]);
            return parent[x];
        }

        public boolean union(int a, int b) {
            int parentA = find(a);
            int parentB = find(b);
            if (parentA == parentB) {
                return false;
            }
            if (size[parentA] > size[parentB]) {
                parent[parentB] = parentA;
                size[parentA] += size[parentB];
            } else {
                parent[parentA] = parentB;
                size[parentB] += size[parentA];
            }
            return true;
        }
    }

    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        UnionFind uf = new UnionFind(m * n);
        List<Integer> res = new ArrayList<Integer>();
        HashSet<Integer> lands = new HashSet<Integer>();
        int count = 0;
        for (int[] pos : positions) {
            int x = pos[0];
            int y = pos[1];
            if (lands.contains(x * n + y)) {
                res.add(count);
                continue;
            }
            lands.add(x * n + y);
            count++;
            for (int[] dir : dirs) {
                int newX = x + dir[0];
                int newY = y + dir[1];
                if (newX < 0 || newX >= m || newY < 0 || newY >= n || !lands.contains(newX * n + newY)) {
                    continue;
                }
                if (uf.union(x * n + y, newX * n + newY)) {
                    count--;
                }
            }
            res.add(count);
        }
        return res;
    }
}