/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private TreeNode res;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        res = null;
        dfs(root, p, q);
        return res;
    }

    private int dfs(TreeNode cur, TreeNode p, TreeNode q) {
        if (cur == null) {
            return 0;
        }
        int count = 0;
        if (cur == p || cur == q) {
            count++;
        }
        count += dfs(cur.left, p, q);
        count += dfs(cur.right, p, q);
        if (count == 2) {
            res = cur;
            return 0;
        }
        return count;
    }
}