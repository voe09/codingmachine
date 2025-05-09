// O(N) 
class Solution {
    public int countNicePairs(int[] nums) {
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        long count = 0;
        int MOD = (int)Math.pow(10, 9) + 7;
        for (int num : nums) {
            long revNum = rev(num);
            map.put((long)num - revNum, map.getOrDefault((long)num - revNum, 0) + 1);
        }
        for (int value : map.values()) {
            count += (long)value * (value - 1) / 2 % MOD;
            count %= MOD;
        }
        return (int)count;
    }

    private long rev(int num) {
        long res = 0;
        while (num > 0) {
            int d = num % 10;
            num /= 10;
            res = res * 10 + d;
        }
        return res;
    }
}