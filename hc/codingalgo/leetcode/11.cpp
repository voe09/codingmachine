class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int mx = 0;
        while (left < right) {
            mx = max(mx, min(height[left], height[right]) * (right - left));
            height[left] > height[right] ? --right : ++left;
        }
        return mx;
    }
};


// Why height[left] == height[right], we can move either of them
// it doesn't matter, the final answer will either include all of them or none of them. 
// If there is a higher height in the middle, the algorithm 