// Segment tree
// O(N * log max_index)
class Solution {
    class SegTreeNode {
        int start;
        int end;
        int maxBlock;
        int leftMostBlock;
        int rightMostBlock;
        SegTreeNode left;
        SegTreeNode right;

        public SegTreeNode(int start, int end) {
            this.start = start;
            this.end = end;
            leftMostBlock = end;
            rightMostBlock = start;
            maxBlock = 0;
        }
    }

    private void updateSegTree(SegTreeNode node, int obs) {
        if (node.start == node.end) {
            return;
        }
        int mid = (node.end - node.start) / 2 + node.start;
        if (node.left == null) {
            node.left = new SegTreeNode(node.start, mid);
            node.right = new SegTreeNode(mid + 1, node.end);
        }
        if (mid >= obs) {
            node.left.rightMostBlock = Math.max(node.left.rightMostBlock, obs);
            updateSegTree(node.left, obs);
        } else {
            node.right.leftMostBlock = Math.min(node.right.leftMostBlock, obs);
            updateSegTree(node.right, obs);
        }
        node.maxBlock = Math.max(Math.max(node.left.maxBlock, node.right.maxBlock), node.right.leftMostBlock - node.left.rightMostBlock);
    }

    public int getMaxBlock(SegTreeNode node, int start, int end) {
        if (start >= node.end || end <= node.start) {
            return 0;
        }
        if (start == node.start && end == node.end) {
            return node.maxBlock;
        }
        int mid = (node.end - node.start) / 2 + node.start;
        if (node.left == null) {
            node.left = new SegTreeNode(node.start, mid);
            node.right = new SegTreeNode(mid + 1, node.end);
        }
        int left = getMaxBlock(node.left, node.start, Math.min(mid, end));
        int right = getMaxBlock(node.right, mid + 1, end);
        return Math.max(Math.max(left, right), Math.min(node.right.leftMostBlock, end) - node.left.rightMostBlock);
    }

    public List<Boolean> getResults(int[][] queries) {
        final int MAX_X = 5 * (int)Math.pow(10, 5);
        List<Boolean> res = new ArrayList<Boolean>();
        SegTreeNode root = new SegTreeNode(0, MAX_X);
        for (int[] query : queries) {
            if (query[0] == 1) {
                updateSegTree(root, query[1]);
            } else {
                int size = getMaxBlock(root, 0, query[1]);
                res.add(size >= query[2]);
            }
        }
        return res;
    }
}