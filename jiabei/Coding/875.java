class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1;
        int right = 0;
        for (int p : piles) {
            right = Math.max(p, right);
        }
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (canFinish(piles, h, mid)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private boolean canFinish(int[] piles, int h, int k) {
        int time = 0;
        for (int p : piles) {
            time += p % k == 0 ? p / k : (p / k) + 1;
            if (time > h) return false;
        }
        return time <= h;
    }
}