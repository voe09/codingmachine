class Solution:
    def queryResults(self, limit: int, queries: List[List[int]]) -> List[int]:
        ball_to_color = {}
        color_count = defaultdict(int)
        res = []

        for ball, new_color in queries:
            if ball in ball_to_color:
                old_color = ball_to_color[ball]
                if old_color != new_color:
                    color_count[old_color] -= 1
                    if color_count[old_color] == 0:
                        del color_count[old_color]
                else:
                    res.append(len(color_count))
                    continue

            ball_to_color[ball] = new_color
            color_count[new_color] += 1
            res.append(len(color_count))

        return res