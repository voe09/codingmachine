# version: mmm.nnn.pp

def valid(version: str) -> bool:
    return version <= "103.001.01"

def find_the_last_version(versions: list[str]):
    def sort_key(version: str) -> tuple[int, int, int]:
        major, minor, patch = version.split('.')
        return (int(major), int(minor), int(patch))

    versions = sorted(versions, key=sort_key)

    left, right = 0, len(versions)
    while left < right:
        mid = left + (right - left) // 2
        if not valid(versions[mid]):
            right = mid
        else:
            left = mid + 1
    
    return "" if left - 1 < 0 else versions[left - 1]


versions = ["103.003.02", "103.003.03", "203.003.02", "123.521.01", "100.001.01", "102.001.01"]

print(find_the_last_version(versions))
