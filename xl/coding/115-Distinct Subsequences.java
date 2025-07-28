class Solution {
    private int[][] mem;
    private int len1;
    private int len2;

    public int numDistinct(String s, String t) {
        len1 = s.length();
        len2 = t.length();
        mem = new int[len1][len2];
        for (int[] row : mem) {
            Arrays.fill(row, -1);
        }
        return getDistCount(s, t, 0, 0);
    }

    private int getDistCount(String s, String t, int x, int y) {
        if (x == len1 && y < len2) {
            return 0;
        }
        if (y == len2) {
            return 1;
        }
        if (mem[x][y] != -1) {
            return mem[x][y];
        }
        if (s.charAt(x) != t.charAt(y)) {
            mem[x][y] = getDistCount(s, t, x + 1, y);
        } else {
            mem[x][y] = getDistCount(s, t, x + 1, y + 1) + getDistCount(s, t, x + 1, y);
        }
        return mem[x][y];
    }
}