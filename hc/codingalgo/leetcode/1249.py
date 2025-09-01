class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        left: int = 0
        right: int = 0
        res: str = ""

        for c in s:
            if c == ")":
                right += 1
        
        for c in s:
            if c == "(":
                if left == right:
                    continue
                left += 1
            elif c == ")":
                right -= 1
                if left == 0:
                    continue
                left -= 1
            res += c
        
        return res

            
        