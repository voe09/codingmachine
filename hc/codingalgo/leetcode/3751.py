class Solution:
    def totalWaviness(self, num1: int, num2: int) -> int:
        MAX = min(100001, num2 + 1)
        dp = [0] * MAX
        pref = [0] * MAX

        for i in range(101, MAX):
            d1 = i % 10
            d2 = (i // 10) % 10
            d3 = (i // 100) % 10
            wave = (d2 > max(d1, d3)) or (d2 < min(d1, d3))

            dp[i] = dp[i // 10] + wave
            pref[i] = pref[i-1] + dp[i]
        return pref[num2] - pref[num1 - 1]