class TreeNode:

    def __init__(self):
        self.childs = {}
        self.word = None

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TreeNode()

        for word in words:
            curr = root
            for c in word:
                if c not in curr.childs:
                    curr.childs[c] = TreeNode()
                curr = curr.childs[c]
            curr.word = word
        
        m, n = len(board), len(board[0])
        ans = []
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        def dfs(row: int, col: int, node: TreeNode):
            if node.word is not None:
                ans.append(node.word)
                node.word = None # reduce dup
            
            for dir in dirs:
                x = row + dir[0]
                y = col + dir[1]
                if x < 0 or x >= m or y < 0 or y >= n or board[x][y] not in node.childs:
                    continue
                
                origin = board[x][y]
                board[x][y] = "#"
                dfs(x, y, node.childs[origin])
                board[x][y] = origin

        for i in range(m):
            for j in range(n):
                if board[i][j] in root.childs:
                    origin = board[i][j]
                    board[i][j] = "#"
                    dfs(i, j, root.childs[origin])
                    board[i][j] = origin

        return ans


        