class DetectSquares {
public:
    DetectSquares() {
        
    }
    
    void add(vector<int> point) {
        space[point[0]][point[1]]++;
        
    }
    
    int count(vector<int> point) {
        int x = point[0], y = point[1];

        int res = 0;
        for (auto &p : space) {
            int x1 = p.first;
            if (x == x1) continue;
            for (auto &count : p.second) {
                int y1 = count.first, cnt = count.second;
                if (abs(x - x1) == abs(y - y1)) {
                    res += cnt * space[x][y1] * space[x1][y];
                }
            }
        }
        return res;
    }

    unordered_map<int, unordered_map<int, int>> space;
};

/**
 * Your DetectSquares object will be instantiated and called as such:
 * DetectSquares* obj = new DetectSquares();
 * obj->add(point);
 * int param_2 = obj->count(point);
 */