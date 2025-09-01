struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

class Traverser {
    vector<vector<int>> visit(TreeNode* root) {
        unordered_map<int, vector<int>> m;
        int left = 0, right = 0;
        helper(root, 0, left, right, m);
        vector<vector<int>> res;
        for (int i = left; i <= right; i++) {
            if (m.count(i)) {
                res.push_back(m[i]);
            }
        }
        return res;
    }

    void helper(TreeNode *node, int order, int &left, int &right, unordered_map<int, vector<int>> &m) {
        if (!node) return;
        left = min(order, left), right = max(order, right);
        m[order].push_back(node->val);
        helper(node->left, order - 1, left, right, m);
        helper(node->right, order + 1, left, right, m);
    }
};

int main() {
    TreeNode *node = 
};

