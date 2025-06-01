// O(N log M)
// N - piles.length
// M - max piles[i]
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int max = 0;
        for (int count : piles) {
            max = Math.max(max, count);
        }
        int start = 1, end = max;
        int mid = 0;
        while (start <= end) {
            mid = (end - start) / 2 + start;
            long time = getTimeCost(piles, mid);
            if (time <= h) {
                end = mid - 1;
            } else if (time > h) {
                start = mid + 1;
            }
        }
        return start;
    }
    
    private long getTimeCost(int[] piles, int k) {
        long sum = 0;
        for (int count : piles) {
            sum += Math.ceil((double)count / k);
        }
        return sum;
    }
}