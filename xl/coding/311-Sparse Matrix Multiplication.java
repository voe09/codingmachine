// 如果优化：用0多的过一遍去乘 
class Solution {
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        int rows1 = mat1.length, cols2 = mat2[0].length;
        int colrows = mat1[0].length; // = mat2 rows
        int[][] ans = new int[rows1][cols2];
        // if (countZero(mat1) < countZero(mat2)) {
        // iterate mat2
        // }
        for (int row1 = 0; row1 < rows1; row1++) {
            for (int col1 = 0; col1 < colrows; col1++) {
                int row2 = col1;
                if (mat1[row1][col1] == 0) {
                    continue;
                }
                for (int col2 = 0; col2 < cols2; col2++) {
                    ans[row1][col2] += mat1[row1][col1] * mat2[row2][col2];
                }
            }
        }
        return ans;
    }

    // private int countZero(int[][] mat) {
    //     int count = 0;
    //     for (int i = 0; i < mat.length; i++) {
    //         for (int j = 0; j < mat[0].length; j++) {
    //             if (mat[i][j] == 0) {
    //                 count++;
    //             }
    //         }
    //     }
    //     return count;
    // }
}

//////////////////////////////////////////////
// 压缩，节约空间，nt，题目里不一起问了
class Solution {
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        int rows1 = mat1.length, cols2 = mat2[0].length;
        int colrows = mat1[0].length; // = mat2 rows
        int[][] ans = new int[rows1][cols2];

        List<Map<Integer, Integer>> cmat1 = compressMat(mat1);
        List<Map<Integer, Integer>> cmat2 = compressMat(mat2);

        for (int row1 = 0; row1 < cmat1.size(); row1++) {
            for (Map.Entry<Integer, Integer> e1 : cmat1.get(row1).entrySet()) {
                int col1 = e1.getKey();
                int val1 = e1.getValue();
                for (Map.Entry<Integer, Integer> e2 : cmat2.get(col1).entrySet()) {
                    ans[row1][e2.getKey()] += val1 * e2.getValue();
                }
            }
        }

        return ans;
    }

    private List<Map<Integer, Integer>> compressMat(int[][] mat) {
        List<Map<Integer, Integer>> res = new ArrayList<Map<Integer, Integer>>();
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                int val = mat[i][j];
                if (res.size() == i) {
                    res.add(new HashMap<Integer, Integer>());
                }
                if (val == 0) {
                    continue;
                }
                res.get(i).put(j, val);
            }
        }

        return res;
    }
}