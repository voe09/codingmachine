class Solution:
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        m, n = len(grid), len(grid[0])

        res = [[0] * (n - 2) for _ in range(m-2)]

        for i in range(m - 2):
            for j in range(n - 2):
                mx = 0
                for x in range(i, i+3):
                    for y in range(j, j+3):
                        mx = max(mx, grid[x][y])
                res[i][j] = mx
        
        return res
        