import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque

# class BankAccount:
#     def __init__(self, balance=0):
#         self.balance = balance
#         self._lock = threading.Lock()

#     def deposit(self, amount):
#         with self._lock:
#             new_balance = self.balance + amount  # Read
#             time.sleep(0.1)                      # Delay
#             self.balance = new_balance           # Write


class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self.queue = deque([])
        self._lock = threading.Condition()
        self._worker = threading.Thread(target=self._process)
        self._worker.start()

    def _process(self):
        while True:
            with self._lock:
                while not self.queue:
                    self._lock.wait()
            
                deposit, future = self.queue.popleft()

            self.balance += deposit
            future.set_result(self.balance)

    def deposit(self, amount):
        future = Future()
        with self._lock:
            self.queue.append((amount, future))
            self._lock.notify()
        
        return future
    

def main():
    account = BankAccount(0)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit the 'deposit' task to the pool
        # pool_f is the "receipt" for dropping off the check
        pool_f1 = executor.submit(account.deposit, 500)
        pool_f2 = executor.submit(account.deposit, 700)

        # To get the final balance, we get the Future returned by deposit()
        bank_future1 = pool_f1.result() 
        bank_future2 = pool_f2.result()

        print(f"Transaction 1 Finished: {bank_future1.result()}")
        print(f"Transaction 2 Finished: {bank_future2.result()}")

    print(f"Final Balance: {account.balance}")

if __name__ == "__main__":
    main()