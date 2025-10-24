# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left = 1
        right = n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left



# Follow up
class VersionControl:

    def __init__(self, bad_version: str):
        self.bad_version = tuple(int(v) for v in bad_version.split('.'))

    def isBadVersion(self, version: str) -> bool:
        version_num = tuple(int(v) for v in version.split('.'))
        is_bad = version_num >= self.bad_version
        return is_bad

checker = VersionControl("2.1.9")

def firstBadVersion(latest_version: str) -> str:
    max_m, max_n, max_p = [int(v) for v in latest_version.split('.')]

    left = 1
    right = max_m
    while left < right:
        mid = left + (right - left) // 2
        if checker.isBadVersion(f"{mid}.9.9"):
            right = mid
        else:
            left = mid + 1
    bad_m = left

    left = 1
    right = 9
    while left < right:
        mid = left + (right - left) // 2
        if checker.isBadVersion(f"{bad_m}.{mid}.9"):
            right = mid
        else:
            left = mid + 1
    bad_n = left

    left = 1
    right = 9
    while left < right:
        mid = left + (right - left) // 2
        if checker.isBadVersion(f"{bad_m}.{bad_n}.{mid}"):
            right = mid
        else:
            left = mid + 1
    bad_p = left

    return f"{bad_m}.{bad_n}.{bad_p}"


print(firstBadVersion("3.1.0"))