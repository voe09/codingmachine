class Solution {
    public int maximalRectangle(char[][] matrix) {
        // rows, cols > 0
        int rows = matrix.length, cols = matrix[0].length;
        int[] hei = new int[cols];
        int[] leftB = new int[cols];
        int[] rightB = new int[cols];
        int res = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '1') {
                    hei[j]++;
                } else {
                    hei[j] = 0;
                }
            }
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(-1);
            for (int j = 0; j < cols; j++) {
                while (stack.size() > 1 && hei[stack.peek()] >= hei[j]) {
                    stack.pop();
                }
                leftB[j] = stack.peek();
                stack.push(j);
            }
            stack.clear();
            stack.push(cols);
            for (int j = cols - 1; j >= 0; j--) {
                while (stack.size() > 1 && hei[stack.peek()] >= hei[j]) {
                    stack.pop();
                }
                rightB[j] = stack.peek();
                stack.push(j);
            }
            for (int j = 0; j < cols; j++) {
                int curArea = hei[j] * (rightB[j] - leftB[j] - 1);
                res = Math.max(res, curArea);
            }
        }
        return res;
    }
}