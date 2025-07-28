class Solution {
    public int nextGreaterElement(int n) {
        StringBuilder sb = new StringBuilder(String.valueOf(n));
        int len = sb.length();
        int toSwap = -1;
        for (int i = len - 1; i > 0; i--) {
            if (sb.charAt(i - 1) < sb.charAt(i)) {
                toSwap = i - 1;
                for (int j = len - 1; j > toSwap; j--) {
                    if (sb.charAt(j) > sb.charAt(toSwap)) {
                        char tmp = sb.charAt(j);
                        sb.setCharAt(j, sb.charAt(toSwap));
                        sb.setCharAt(toSwap, tmp);
                        reverse(sb, toSwap + 1, len - 1);
                        break;
                    }
                }
                break;
            }
        }
        if (toSwap == -1) {
            return -1;
        }
        long res = Long.valueOf(sb.toString());
        return res > Integer.MAX_VALUE ? -1 : (int)res;
    }

    private void reverse(StringBuilder sb, int start, int end) {
        while (start < end) {
            char tmp = sb.charAt(start);
            sb.setCharAt(start, sb.charAt(end));
            sb.setCharAt(end, tmp);
            start++;
            end--;
        }
    }
}