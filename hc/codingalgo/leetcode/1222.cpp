class Solution {
public:
    vector<vector<int>> queensAttacktheKing(vector<vector<int>>& queens,
                                            vector<int>& king) {
        unordered_map<int, unordered_set<int>> m;
        for (vector<int>& q : queens) {
            m[q[0]].insert(q[1]);
        }

        vector<vector<int>> res;
        vector<vector<int>> dirs = {{0, 1}, {0, -1}, {1, 0},  {-1, 0},
                                    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (auto& dir : dirs) {
            bool seen = false;
            for (int x = king[0] + dir[0], y = king[1] + dir[1];
                 x >= 0 && x < 8 && y >= 0 && y < 8; x += dir[0], y += dir[1]) {
                if (m.count(x) && m[x].count(y) && !seen) {
                    res.push_back({x, y});
                    seen = true;
                }
            }
        }

        return res;
    }
};