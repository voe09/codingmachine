// O(N log N) - TreeMap
//////////////////////////////////

// O(N) - monotonic queue
class Solution {
    public int longestSubarray(int[] nums, int limit) {
        int l = 0, r = 0, len = nums.length;
        int res= 1;
        Deque<Integer> minDq = new ArrayDeque<Integer>();
        Deque<Integer> maxDq = new ArrayDeque<Integer>();
        while (r < len) {
            int cur = nums[r];
            while (!minDq.isEmpty() && minDq.peekLast() > cur) {
                minDq.pollLast();
            }
            minDq.offerLast(cur);

            while (!maxDq.isEmpty() && maxDq.peekLast() < cur) {
                maxDq.pollLast();
            }
            maxDq.offerLast(cur);

            while (maxDq.peekFirst() - minDq.peekFirst() > limit) {
            // 2 deques always contain one num at least
                if (maxDq.peekFirst() == nums[l]) {
                    maxDq.pollFirst();
                }
                if (minDq.peekFirst() == nums[l]) {
                    minDq.pollFirst();
                }
                l++;
            }
            res = Math.max(res, r - l + 1);
            r++;
        }
        return res;
    }
}