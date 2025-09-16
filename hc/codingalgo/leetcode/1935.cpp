class Solution {
public:
    int canBeTypedWords(string text, string brokenLetters) {
        int res = 0;
        unordered_set<char> broken(brokenLetters.begin(), brokenLetters.end());
        for (int i = 0; i < text.size(); i++) {
            bool isBroken = false;
            while (i < text.size() && text[i] != ' ') {
                if (broken.count(text[i])) {
                    isBroken = true;
                }
                i++;
            }
            if (!isBroken) res++;
        }
        return res;
    }
};