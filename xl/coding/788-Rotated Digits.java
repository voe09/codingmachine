class Solution {
    public int rotatedDigits(int N) {
        char[] A = String.valueOf(N).toCharArray();
        int K = A.length;

        int[][][] memo = new int[K+1][2][2];
        memo[K][0][1] = memo[K][1][1] = 1;
        for (int i = K - 1; i >= 0; --i) {
            for (int eqf = 0; eqf <= 1; ++eqf)
                for (int invf = 0; invf <= 1; ++invf) {
                    int ans = 0;
                    for (char d = '0'; d <= (eqf == 1 ? A[i] : '9'); ++d) {
                        if (d == '3' || d == '4' || d == '7') continue;
                        if (invf == 1) {
                            ans += memo[i+1][d == A[i] ? eqf : 0][1];
                        } else {
                            boolean invo = (d == '2' || d == '5' || d == '6' || d == '9');
                            ans += memo[i+1][d == A[i] ? eqf : 0][invo ? 1 : 0];
                        }
                    }
                    memo[i][eqf][invf] = ans;
                }
        }

        return memo[0][1][0];
    }

}