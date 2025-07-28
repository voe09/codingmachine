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
    private HashMap<Pair<Integer, Integer>, List<TreeNode>> mem;

    public List<TreeNode> generateTrees(int n) {
        mem = new HashMap<>();
        return dfs(1, n);
    }

    private List<TreeNode> dfs(int leftB, int rightB) {
        List<TreeNode> res = new ArrayList<TreeNode>();
        if (leftB > rightB) {
            res.add(null);
            return res;
        }
        if (leftB == rightB) {
            res.add(new TreeNode(leftB));
            return res;
        }
        if (mem.containsKey(new Pair<>(leftB, rightB))) {
            return mem.get(new Pair<>(leftB, rightB));
        }
        for (int cur = leftB; cur <= rightB; cur++) {
            List<TreeNode> leftChildren = dfs(leftB, cur - 1);
            List<TreeNode> rightChildren = dfs(cur + 1, rightB);
            for (TreeNode left : leftChildren) {
                for (TreeNode right : rightChildren) {
                    TreeNode curNode = new TreeNode(cur, left, right);
                    res.add(curNode);
                }
            }
        }
        mem.put(new Pair<>(leftB, rightB), res);
        return res;
    }
}