class Solution {
    public String nearestPalindromic(String n) {
        final int PALIN_COUNT = 5;
        int len = n.length();
        boolean isOdd = len % 2 == 1;
        long selfPalin = Long.parseLong(n);
        long firstHalf = Long.parseLong(n.substring(0, (len + 1) / 2));
        List<Long> palins = new ArrayList<Long>(PALIN_COUNT);
        palins.add((long)Math.pow(10, len - 1) - 1);
        palins.add(getPalin(firstHalf - 1, isOdd));
        palins.add(getPalin(firstHalf, isOdd));
        palins.add(getPalin(firstHalf + 1, isOdd));
        palins.add((long)Math.pow(10, len) + 1);
        long closest = palins.get(0);
        for (long palin : palins) {
            if (palin == selfPalin) {
                continue;
            }
            if (Math.abs(selfPalin - palin) < Math.abs(selfPalin - closest)) {
                closest = palin;
            }
        }
        return String.valueOf(closest);
    }

    private long getPalin(long num, boolean isOdd) {
        long res = num;
        num = isOdd ? num / 10 : num;
        while (num > 0) {
            res = res * 10 + num % 10;
            num /= 10;
        }
        return res;
    }
}