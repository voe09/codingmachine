def max_subgrid_size(grid: list[list[int]], maxsum: int) -> int:
    m, n = len(grid), len(grid[0])
    
    # 1. Build (m+1) x (n+1) prefix sum matrix for easier indexing
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (
                grid[i - 1][j - 1]
                + prefix[i - 1][j]
                + prefix[i][j - 1]
                - prefix[i - 1][j - 1]
            )

    # 2. Function to get sum of subgrid [(r1, c1), (r2, c2)], inclusive
    def get_sum(r1, c1, r2, c2):
        return (
            prefix[r2][c2]
            - prefix[r1 - 1][c2]
            - prefix[r2][c1 - 1]
            + prefix[r1 - 1][c1 - 1]
        )


    # 3. Check if all k×k subgrids have sum ≤ maxsum
    def valid(k):
        for i in range(1, m - k + 2):
            for j in range(1, n - k + 2):
                if get_sum(i, j, i + k - 1, j + k - 1) > maxsum:
                    return False
        return True

    # 4. Binary search for max k
    left, right = 1, min(m, n)
    while left < right:
        mid = left + (right - left) // 2 
        if valid(mid):
            left = mid + 1
        else:
            right = mid

    return left


grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
maxsum = 100

print(max_subgrid_size(grid, maxsum))