// O(M * N)
// 可以一行一行来不用array记 stopBy，省一点空间
class Solution {
    public char[][] rotateTheBox(char[][] boxGrid) {
        int rows = boxGrid.length, cols = boxGrid[0].length;
        int[] stopBy = new int[rows];
        Arrays.fill(stopBy, cols - 1);
        char[][] res = new char[cols][rows];
        for (int j = cols - 1; j >= 0; j--) {
            for (int i = 0; i < rows; i++) {
                char cur = boxGrid[i][j];
                if (cur == '*') {
                    res[j][rows - i - 1] = '*';
                    stopBy[i] = j - 1;
                } else if (cur == '.') {
                    res[j][rows - i - 1] = '.';
                } else {
                // stone #
                    res[j][rows - i - 1] = '.';
                    res[stopBy[i]][rows - i - 1] = '#';
                    stopBy[i]--;
                }
            }
        }
        return res;
    }
}