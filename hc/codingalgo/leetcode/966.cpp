class Solution {
public:
    vector<string> spellchecker(vector<string>& wordlist, vector<string>& queries) {
        unordered_set<string> list;
        for (string &w : wordlist) {
            list.insert(w);
        }


        vector<string> res;
        for (string &q : queries) {
            if (list.count(q)) res.push_back(q);
            else res.push_back(helper(wordlist, q));
        }
        return res;
    }

    bool isUpper(char c) {
        return c >= 'A' && c <= 'Z';
    }

    bool isVowel(char c) {
        if (isUpper(c)) c = 'a' + (c - 'A');
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    bool isCaseInsensitive(string &w, string &q) {
        if (w.size() != q.size()) return false;
        for (int i = 0; i < w.size(); i++) {
            if (w[i] != q[i]) {
                int w_idx = isUpper(w[i]) ? w[i] - 'A' : w[i] - 'a';
                int q_idx = isUpper(q[i]) ? q[i] - 'A' : q[i] - 'a';
                if (w_idx != q_idx) return false;
            } 
        }
        return true;
    }

    bool isVowelErr(string &w, string &q) {
        if (w.size() != q.size()) return false;
        for (int i = 0; i < w.size(); i++) {
            if (w[i] != q[i]) {
                int w_idx = isUpper(w[i]) ? w[i] - 'A' : w[i] - 'a';
                int q_idx = isUpper(q[i]) ? q[i] - 'A' : q[i] - 'a';
                if (w_idx != q_idx & !(isVowel(w[i]) && isVowel(q[i]))) {
                    return false;
                } 
            }
        }
        return true;
    }

    string helper(vector<string> &words, string &q) {
        for (string &w : words) {
            if (isCaseInsensitive(w, q)) return w;
            
        }

        for (string &w: words) {
            if (isVowelErr(w, q)) return w;
        }

        return "";
    }
};


class Solution {
public:
    vector<string> spellchecker(vector<string>& wordlist, vector<string>& queries) {
        unordered_set<string> st;
        unordered_map<string, string> m1, m2;
        for (string w : wordlist) {
            st.insert(w);
            string origin_w = w;
            toLower(w);
            if (!m1.count(w)) m1[w] = origin_w;

            vowelErr(w);
            if (!m2.count(w)) m2[w] = origin_w;
        }

        vector<string> res;
        for (string q : queries) {
            if (st.count(q)) {
                res.push_back(q);
                continue;
            }

            toLower(q);
            if (m1.count(q)) {
                res.push_back(m1[q]);
                continue;
            }

            vowelErr(q);
            if (m2.count(q)) {
                res.push_back(m2[q]);
                continue;
            }

            res.push_back("");
        }

        return res;
    }


    void toLower(string &w) {
        for (int i = 0; i < w.size(); i++) {
            w[i] = isUpper(w[i]) ? 'a' + (w[i] - 'A') : w[i]; 
        }
    }

    void vowelErr(string &w) {
        for (int i = 0; i < w.size(); i++) {
            if (isVowel(w[i])) w[i] = '*';
        }
    }

    bool isUpper(char c) {
        return c >= 'A' && c <= 'Z';
    }

    bool isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
};