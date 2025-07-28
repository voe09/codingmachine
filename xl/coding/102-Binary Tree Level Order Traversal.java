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
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        res.add(new ArrayList<>());
        Queue<TreeNode> q = new ArrayDeque<>();
        q.offer(root);
        int size = q.size();
        while (!q.isEmpty()) {
            TreeNode cur = q.poll();
            size--;
            res.get(res.size() - 1).add(cur.val);
            if (cur.left != null) {
                q.offer(cur.left);
            }
            if (cur.right != null) {
                q.offer(cur.right);
            }
            if (size == 0) {
                size = q.size();
                if (size != 0) {
                    res.add(new ArrayList<>());
                }
            }
        }
        return res;
    }
}