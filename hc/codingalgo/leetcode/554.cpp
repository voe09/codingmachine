class Solution {
public:
    int leastBricks(vector<vector<int>>& wall) {
        unordered_map<long long, int> cnts;
        int m = wall.size();

        int mx = 0;
        for (int i = 0; i < m; i++) {
            long long w = 0;
            for (int j = 0; j < wall[i].size() - 1; j++) {
                w += static_cast<long long>(wall[i][j]);
                cnts[w]++;
                mx = max(mx, cnts[w]);
            }
        }

        return m - mx;
    }
};