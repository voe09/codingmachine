/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    private int res;
    public int maxPathSum(TreeNode root) {
        res = root.val;
        findMax(root);
        return res;
    }

    private int findMax(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = Math.max(0, findMax(node.left));
        int right = Math.max(0, findMax(node.right));
        int sum = node.val + left + right;
        res = Math.max(res, sum);
        
        int ret = node.val + Math.max(left, right);
        return ret > 0 ? ret : 0;
    }
}