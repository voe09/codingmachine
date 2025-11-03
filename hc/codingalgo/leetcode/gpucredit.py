class GPUCredit:

    def __init__(self):
        # cost is negative creates at [timestamp, timestamp]
        self.credits: dict[str, list[tuple[int, int, int]]] = {}

    def addCredit(self, creditID: str, amount: int, timestamp: int, expiration: int):
        if creditID not in self.credits:
            self.credits[creditID] = [(timestamp, timestamp + expiration, amount)]
        
        else:
            self.credits[creditID].append((timestamp, timestamp + expiration, amount))

    def getBalance(self, creditID: str, timestamp: int):
        if creditID not in self.credits:
            return None
        
        credits = self.credits[creditID]
        # find 1st credits whose start time <= timestamp
        # find last credits whose end time >= timestamp
        credits = [c for c in credits if c[0] <= timestamp and c[1] >= timestamp]

        balance = sum([c[2] for c in credits])

        if balance <= 0:
            return None
        return balance

    def useCredit(self, creditID: str, timestamp: int, amount: int) -> bool:
        balance = self.getBalance(creditID, timestamp)
        if balance is None or balance < amount:
            return False

        self.credits[creditID].append((timestamp, 1e10, -amount))
        return True


gpu_credit = GPUCredit()
gpu_credit.addCredit("openai", 100, 10, 30)
gpu_credit.addCredit("gcp", 10, 100, 100)
gpu_credit.addCredit("openai", 100, 20, 30)
print(gpu_credit.getBalance("openai", 10)) # 100
print(gpu_credit.getBalance("openai", 20)) # 200
print(gpu_credit.getBalance("openai", 40)) # 200
print(gpu_credit.getBalance("openai", 50)) # 100
print(gpu_credit.getBalance("openai", 60)) # None
print(gpu_credit.useCredit("openai", 10, 200)) # 100
print(gpu_credit.useCredit("openai", 10, 100)) # 100
print(gpu_credit.getBalance("openai", 10)) # 100
print(gpu_credit.getBalance("openai", 20)) # 200
print(gpu_credit.getBalance("openai", 40)) # 200
print(gpu_credit.getBalance("openai", 50)) # 100
print(gpu_credit.getBalance("openai", 60)) # None



class Credits:
    def __init__(self):
        self.grants = []  # (grant_id, amount, expiration_ts, start_ts)
        self.subtractions = []  # (amount, ts)

    def create_grant(
        self, 
        grant_id: str, 
        amount: int, 
        timestamp: int,
        expiration_timestamp: int, 
    ) -> None:
        self.grants.append([grant_id, amount, timestamp, expiration_timestamp])

    def subtract(self, amount: int, timestamp: int) -> None:
        self.subtractions.append([amount, timestamp])

    def get_balance(self, timestamp: int) -> int:
        relevant_orders = [list(o) for o in self.subtractions if o[1] <= timestamp]

        relevant_grants = [list(g) for g in self.grants if g[2] <= timestamp]

        relevant_grants = sorted(relevant_grants, key = lambda x: x[3])
        relevant_orders = sorted(relevant_orders, key = lambda x: x[1])

        i = 0
        j = 0
        while i < len(relevant_orders) and j < len(relevant_grants):
            amount, ts = relevant_orders[i]
            _, credit, start_ts, end_ts = relevant_grants[j]
            if end_ts < ts:
                j += 1
            else: # end_ts >= ts
                if start_ts > ts:
                    j += 1
                    continue
                if amount <= credit:
                    relevant_grants[j][1] = credit - amount
                    relevant_orders[i][0] = 0
                    i += 1
                else:
                    relevant_grants[j][1] = 0
                    relevant_orders[i][0] = amount - credit
                    j += 1
        
        relevant_grants = [g for g in relevant_grants if g[3] > timestamp]
        balance = sum([g[1] for g in relevant_grants])
        return balance


credits = Credits()
credits.subtract(amount=1, timestamp=30)
credits.create_grant(grant_id="a", amount=1, timestamp=10, expiration_timestamp=100)
print(credits.get_balance(timestamp=10)) # 1
print(credits.get_balance(timestamp=30)) # 0
print(credits.get_balance(timestamp=20)) # 1
print(credits.get_balance(timestamp=20)) # 1

credits = Credits()
credits.create_grant(grant_id="a", amount=3, timestamp=10, expiration_timestamp=60)
assert credits.get_balance(10) == 3
credits.create_grant(grant_id="b", amount=2, timestamp=20, expiration_timestamp=40)
credits.subtract(amount=1, timestamp=30)
credits.subtract(amount=3, timestamp=50)
assert credits.get_balance(10) == 3
assert credits.get_balance(20) == 5
assert credits.get_balance(30) == 4
assert credits.get_balance(40) == 3 # minimize expiring. Subtract expiration from usage
assert credits.get_balance(50) == 0


credits = Credits()
credits.create_grant(grant_id="a", amount=3, timestamp=10, expiration_timestamp=60)
credits.create_grant(grant_id="b", amount=2, timestamp=20, expiration_timestamp=80)
credits.subtract(amount=4, timestamp=30)
assert credits.get_balance(10) == 3
assert credits.get_balance(20) == 5
assert credits.get_balance(30) == 1
assert credits.get_balance(70) == 1


import unittest

from heapq import heappush, heappop

class GPUCredit:

    def __init__(self):
        self.credits = []
        self.subs = []

    def transaction(self, id: str, amount: int, start_ts: int, end_ts: int):
        self.credits.append([id, amount, start_ts, end_ts])

    def substract(self, amount: int, ts: int):
        self.subs.append([amount, ts])

    def get_balance(self, ts: int):
        # get all credits and subs whose start ts is <= ts
        credits = [list(c) for c in self.credits if c[2] <= ts]
        subs = [list(s) for s in self.subs if s[1] <= ts]

        # keep a priority queue, expired first credits will be the top
        heap = []
        i = 0
        for amount, sub_ts in subs:
            while i < len(credits) and credits[i][2] <= sub_ts:
                _, credit, _, end_ts = credits[i]
                heappush(heap, (end_ts, credit))
                i += 1
    
            while len(heap) > 0 and heap[0][0] <= sub_ts:
                heappop(heap) # remove credit whose expiration_ts <= sub_ts

            while len(heap) > 0 and amount > 0:
                end_ts, credit = heappop(heap)
                if credit <= amount:
                    amount -= credit
                    credit = 0
                else:
                    credit -= amount
                    amount = 0
                    heappush(heap, (end_ts, credit))
            
        
        while i < len(credits):
            heappush(heap, (credits[i][3], credits[i][1]))
            i += 1

        while len(heap) > 0 and heap[0][0] <= ts:
            heappop(heap)

        balance = 0
        for _, credit in heap:
            balance += credit
        
        return balance



class TestGPUCredit(unittest.TestCase):

    def test_get_balance(self):
        credit = GPUCredit()
        credit.transaction("A", 10, 0, 10)
        credit.transaction("B", 10, 0, 100)
        self.assertEqual(credit.get_balance(5), 20)
        credit.substract(9, 9)
        self.assertEqual(credit.get_balance(9), 11)
        self.assertEqual(credit.get_balance(10), 10)


unittest.main()
