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



