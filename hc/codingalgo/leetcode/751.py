import math

def ipToCIDR(ip: str, n: int):
    res = []
    x = 0

    # convert IP to a 32bit int
    for part in ip.split('.'):
        x = x * 256 + int(part)
    
    while n > 0:
        step = x & -x
        while step > n:
            step //= 2
        res.append(convert(x, step))
        x += step
        n -= step
    return res


def convert(x: int, step: int) -> str:
    return "{}.{},{},{}".format(
        (x >> 24) & 255, 
        (x >> 16) & 255,
        (x >> 8) & 255,
        x & 255,
    ) + "/" + str(32 - int(math.log2(step)))