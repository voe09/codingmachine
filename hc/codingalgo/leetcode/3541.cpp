class Solution {
public:
    int maxFreqSum(string s) {
        vector<int> cnts(26, 0);
        int mxV = 0, mxNonV = 0;
        for (char c : s) {
            cnts[c - 'a']++;
            if (isVowel(c)) {
                mxV = max(mxV, cnts[c-'a']);
            } else {
                mxNonV = max(mxNonV, cnts[c-'a']);
            }
        }

        return mxV + mxNonV;
    }

    bool isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
};