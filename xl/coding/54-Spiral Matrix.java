class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        // rows > 0, cols > 0
        int rows = matrix.length, cols = matrix[0].length;
        List<Integer> res = new ArrayList<>();
        int[][] dirs = new int[][] {
            { 0,  1},
            { 1,  0},
            { 0, -1},
            {-1,  0}
        };
        int[] borders = new int[] { cols, rows, -1, 0 };
        int idx = 0, count = 0;
        int r = 0, c = 0;
        while (count < rows * cols) {
            res.add(matrix[r][c]);
            r += dirs[idx][0];
            c += dirs[idx][1];
            if (idx % 2 == 0 && c == borders[idx]) {
                c -= dirs[idx][1];
                borders[idx] -= dirs[idx][1];
                idx = (idx + 1) % 4;
                r += dirs[idx][0];
                c += dirs[idx][1];
            } else if (idx % 2 == 1 && r == borders[idx]) {
                r -= dirs[idx][0];
                borders[idx] -= dirs[idx][0];
                idx = (idx + 1) % 4;
                r += dirs[idx][0];
                c += dirs[idx][1];
            }
            count++;
        }
        return res;
    }
}