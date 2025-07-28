class Solution {
    public int reverse(int x) {
        int res = 0;
        int sign = x >= 0 ? 1 : -1;
        while (x != 0) {
            if (res > Integer.MAX_VALUE / 10 || res < Integer.MIN_VALUE / 10) {
                return 0;
            }
            res *= 10;
            res += x % 10;
            // if intput is not limited to int
            // if (res > 0 && sign < 0 || (res < 0 && sign > 0)) {
            //     return 0;
            // }
            x /= 10;
        }
        return res;
    }
}