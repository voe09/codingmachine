class Solution {
public:
    vector<int> replaceNonCoprimes(vector<int>& nums) {
        stack<int> stk;
        for (int i = 0; i < nums.size(); i++) {
            int t = nums[i];
            while (!stk.empty() && !coprime(stk.top(), t)) {
                int p = stk.top();
                stk.pop();
                t = lcm(t, p);
            }
            stk.push(t);
        }

        vector<int> res;
        while (!stk.empty()) {
            res.push_back(stk.top());
            stk.pop();
        }
        reverse(res.begin(), res.end());
        return res;
    }

    bool coprime(int a, int b) {
        if (gcd(a, b) == 1) return true;
        return false;
    }

    int gcd(int a, int b) {
        if (a < b) return gcd(b, a);
        return a % b == 0 ? b : gcd(b, a % b);
    }

    int lcm(int a, int b) {
        int c = gcd(a, b);
        return a * (b / c);
    }
};