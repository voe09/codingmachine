# from collections import deque

# class MockLLM:
#     def generate(self, batch_prefix: list[str]) -> list[str]:
#         # Simulating distinct token generation for each request in the batch
#         return [f"t{i+1}_" for i in range(len(batch_prefix))]

# class Request:
#     def __init__(self, request_id: str, prefix: str, max_token: int, stop_token: str):
#         self.request_id = request_id
#         self.res = prefix
#         self.max_token = max_token
#         self.stop_token = stop_token
#         self.finish = False
#         self.generated_count = 0 # Track tokens, not characters

#     def add_token(self, token: str):
#         self.res += token
#         self.generated_count += 1
#         # Termination condition logic
#         if self.generated_count >= self.max_token or token == self.stop_token:
#             self.finish = True

# class InferenceEngine:
#     def __init__(self, batch_size: int):
#         self.batch_size = batch_size
#         self.tasks = deque([])
#         self.slots = [None] * batch_size
#         self.availables = deque(list(range(batch_size)))
#         self.occupied_slots = {} # request_id -> slot_idx for faster lookup
#         self.model = MockLLM()
    
#     def add_request(self, request: Request):
#         self.tasks.append(request)

#     def _fill_slots(self):
#         while self.tasks and self.availables:
#             req = self.tasks.popleft()
#             slot_idx = self.availables.popleft()
#             self.slots[slot_idx] = req
#             self.occupied_slots[req.request_id] = slot_idx
    
#     def run(self):
#         # Continue if there are tasks waiting OR slots are being worked on
#         while self.tasks or len(self.availables) < self.batch_size:
#             self._fill_slots()

#             # Identify which slots currently have an active request
#             active_indices = [i for i, req in enumerate(self.slots) if req is not None]
#             if not active_indices:
#                 break

#             # Step 2: Iterative Decoding
#             batch_inputs = [self.slots[i].res for i in active_indices]
#             next_tokens = self.model.generate(batch_inputs)

#             # Step 3: Update and Recycle
#             for i, next_token in enumerate(next_tokens):
#                 slot_idx = active_indices[i]
#                 req = self.slots[slot_idx]
                
#                 req.add_token(next_token)
                
#                 if req.finish:
#                     print(f"Finished: {req.request_id} | Result: {req.res}")
#                     # Free resources
#                     self.slots[slot_idx] = None
#                     self.availables.append(slot_idx)
#                     del self.occupied_slots[req.request_id]

# # --- Test Case ---
# engine = InferenceEngine(batch_size=2)
# engine.add_request(Request("REQ-A", "Start", max_token=5, stop_token="STOP"))
# engine.add_request(Request("REQ-B", "Begin", max_token=2, stop_token="STOP"))
# engine.add_request(Request("REQ-C", "Go", max_token=3, stop_token="STOP"))

# engine.run()

# import asyncio
# from collections import deque

# class MockLLM:
#     def __init__(self):
#         self.counter = 0

#     async def generate(self, batch_prefix: list[str]) -> list[str]:
#         res = [f"t{self.counter}_" for _ in range(len(batch_prefix))]
#         self.counter += 1
#         return res
    

# class Request:

#     def __init__(self, request_id: str, prompt: str, max_token: int, stop_token: str):
#         self.id = request_id
#         self.res = prompt
#         self.max_token = max_token
#         self.stop_token = stop_token
#         self.generated_count = 0
#         self.finished_event = asyncio.Event()

#     def add(self, token: str):
#         self.res += token
#         self.generated_count += 1
#         if self.generated_count >= self.max_token or token == self.stop_token:
#             return True
#         return False
    

# class InferenceEngine:

#     def __init__(self, batch_size: int):
#         self.bz = batch_size
#         self.waiting_queue = deque([])
#         self.slots = [None] * batch_size
#         self.model = MockLLM()
#         self.is_running = False

#     async def infer(self, prompt: str, max_tokens=10, stop_token="<EOS>"):
#         req_id = f"req_{id(prompt)}"
#         request = Request(request_id=req_id, prompt=prompt, max_token=max_tokens, stop_token=stop_token)

#         self.waiting_queue.append(request)

#         if not self.is_running:
#             asyncio.create_task(self._engine_loop())
#             self.is_running = True
        
#         await request.finished_event.wait()
#         return request.res
    

#     async def _engine_loop(self):
#         while True:
#             for i in range(self.bz):
#                 if self.slots[i] is None and self.waiting_queue:
#                     self.slots[i] = self.waiting_queue.popleft()

#             active_req = [(i, r) for i, r in enumerate(self.slots) if r is not None]
#             if not active_req and not self.waiting_queue:
#                 self.is_running = False
#                 break

#             batch_inputs = [r.res for _, r in active_req]
#             next_tokens = await self.model.generate(batch_inputs)
#             for (i, req), next_token in zip(active_req, next_tokens):
#                 if req.add(next_token):
#                     req.finished_event.set()
#                     self.slots[i] = None


# async def main():
#     engine = InferenceEngine(batch_size=2)
    
#     # Simulate multiple concurrent users
#     prompts = ["Tell me a joke", "Explain Quantum", "Write Python", "Hello world"]
#     tasks = [engine.infer(p) for p in prompts]
    
#     results = await asyncio.gather(*tasks)
#     for res in results:
#         print(f"Final Result: {res}")

# if __name__ == "__main__":
#     asyncio.run(main())

# import asyncio
# from collections import deque

# class MockLLM:

#     def __init__(self):
#         self.counter = 0

#     async def generate(self, batch: list[str]) -> list[str]:
#         tokens = [f"_token_{self.counter}" for _ in batch]
#         self.counter += 1
#         return tokens


# class Request:

#     def __init__(self, id: str, prefix: str, max_token: int, stop_token: str):
#         self.id = id
#         self.res = prefix
#         self.counter = 0
#         self.max_token = max_token
#         self.stop_token = stop_token
#         self.done_event = asyncio.Event()

#     def add(self, token: str) -> bool:
#         self.res += token
#         self.counter += 1
#         if self.counter == self.max_token or token == self.stop_token:
#             self.done_event.set()
#             return True
#         return False
    

# class InferenceEngine:

#     def __init__(self, batch_size: int, max_wait_ms: float = 10.0):
#         self.bz = batch_size
#         self.max_wait_sec = max_wait_ms / 1000
#         self.queue = deque([])
#         self.slots = [None] * batch_size
#         self.empty = deque(list(range(batch_size)))
#         self.occupy = {}
#         self.model = MockLLM()
#         self._running = False

#     async def infer(self, req: Request) -> str:
#         self.queue.append(req)
#         if not self._running:
#             asyncio.create_task(self._engine_loop())
#             self._running = True

#         await req.done_event.wait()
#         return req.res

#     async def _engine_loop(self):
#         while True:
#             # prefix slots
#             while self.queue and self.empty:
#                 res = self.queue.popleft()
#                 slot = self.empty.popleft()
#                 self.slots[slot] = res
#                 self.occupy[res.id] = slot
            
#             if not self.queue and not self.occupy:
#                 self._running = False
#                 break

#             reqs = [r for r in self.slots if r is not None]
#             prefix = [r.res for r in reqs]
#             tokens = await self.model.generate(prefix)
#             for req, token in zip(reqs, tokens):
#                 if req.add(token):
#                     slot = self.occupy.pop(req.id)
#                     self.slots[slot] = None
#                     self.empty.append(slot)


# async def main():
#     engine = InferenceEngine(batch_size=2)
#     requests = [
#         Request("1", "New", max_token=2, stop_token="<EOS>"),
#         Request("2", "Year", max_token=5, stop_token="<EOS>"),
#         Request("3", "Eve", max_token=3, stop_token="<EOS>"),
#     ]
#     tasks = [engine.infer(req) for req in requests]
#     results = await asyncio.gather(*tasks)
#     for res in results:
#         print(res)

# if __name__ == "__main__":
#     asyncio.run(main())

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque


class MockLLM:

    def __init__(self):
        self.counter = 0

    def generate(self, batch: list[str]) -> list[str]:
        tokens = [f"_token_{self.counter}" for _ in batch]
        self.counter += 1
        return tokens
    
class Request:

    def __init__(self, id: str, prefix: str, max_token: int, stop_token: str):
        self.id = id
        self.res = prefix
        self.counter = 0
        self.max_token = max_token
        self.stop_token = stop_token
        self.done_event = threading.Event()

    def add(self, token: str) -> bool:
        self.res += token
        self.counter += 1
        if self.counter == self.max_token or token == self.stop_token:
            self.done_event.set()
            return True
        return False
    
class InferencEngine:

    def __init__(self, batch_size: int, max_wait_ms: float = 10.0):
        self.bz = batch_size
        self.max_wait_sec = max_wait_ms / 1000
        self.queue = deque([])
        self.slots = [None] * batch_size
        self.empty = deque(list(range(batch_size)))
        self.occupy = {}
        self.model = MockLLM()
        self._has_work = threading.Condition() # built on top of Lock

        self.engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self.engine_thread.start()

    def infer(self, req: Request) -> str:
        with self._has_work:
            self.queue.append(req)
            self._has_work.notify()
        
        req.done_event.wait()
        return req.res

    def _engine_loop(self):
        while True:
            with self._has_work:
                while not self.queue and not self.occupy:
                    self._has_work.wait()

                if self.empty and not self.queue:
                    self._has_work.wait(timeout=self.max_wait_sec)

                while self.queue and self.empty:
                    req = self.queue.popleft()
                    slot = self.empty.popleft()
                    self.slots[slot] = req
                    self.occupy[req.id] = slot
            
            reqs = [r for r in self.slots if r is not None]
            if not reqs:
                continue
        
            prefixes = [r.res for r in reqs]
            tokens = self.model.generate(prefixes)

            with self._has_work:
                for req, token in zip(reqs, tokens):
                    if req.add(token):
                        slot = self.occupy.pop(req.id)
                        self.slots[slot] = None
                        self.empty.append(slot)

            
def main():
    engine = InferencEngine(batch_size=2)
    requests = [
        Request("1", "New", max_token=2, stop_token="<EOS>"),
        Request("2", "Year", max_token=5, stop_token="<EOS>"),
        Request("3", "Eve", max_token=3, stop_token="<EOS>"),
    ]
    
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(engine.infer, req) for req in requests]
        for future in futures:
            print(future.result())

if __name__ == "__main__":
    main()