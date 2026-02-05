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

import asyncio
from collections import deque

class MockLLM:
    def __init__(self):
        self.counter = 0

    async def generate(self, batch_prefix: list[str]) -> list[str]:
        res = [f"t{self.counter}_" for _ in range(len(batch_prefix))]
        self.counter += 1
        return res
    

class Request:

    def __init__(self, request_id: str, prompt: str, max_token: int, stop_token: str):
        self.id = request_id
        self.res = prompt
        self.max_token = max_token
        self.stop_token = stop_token
        self.generated_count = 0
        self.finished_event = asyncio.Event()

    def add(self, token: str):
        self.res += token
        self.generated_count += 1
        if self.generated_count >= self.max_token or token == self.stop_token:
            return True
        return False
    

class InferenceEngine:

    def __init__(self, batch_size: int):
        self.bz = batch_size
        self.waiting_queue = deque([])
        self.slots = [None] * batch_size
        self.model = MockLLM()
        self.is_running = False

    async def infer(self, prompt: str, max_tokens=10, stop_token="<EOS>"):
        req_id = f"req_{id(prompt)}"
        request = Request(request_id=req_id, prompt=prompt, max_token=max_tokens, stop_token=stop_token)

        self.waiting_queue.append(request)

        if not self.is_running:
            asyncio.create_task(self._engine_loop())
            self.is_running = True
        
        await request.finished_event.wait()
        return request.res
    

    async def _engine_loop(self):
        while True:
            for i in range(self.bz):
                if self.slots[i] is None and self.waiting_queue:
                    self.slots[i] = self.waiting_queue.popleft()

            active_req = [(i, r) for i, r in enumerate(self.slots) if r is not None]
            if not active_req and not self.waiting_queue:
                self.is_running = False
                break

            batch_inputs = [r.res for _, r in active_req]
            next_tokens = await self.model.generate(batch_inputs)
            for (i, req), next_token in zip(active_req, next_tokens):
                if req.add(next_token):
                    req.finished_event.set()
                    self.slots[i] = None


async def main():
    engine = InferenceEngine(batch_size=2)
    
    # Simulate multiple concurrent users
    prompts = ["Tell me a joke", "Explain Quantum", "Write Python", "Hello world"]
    tasks = [engine.infer(p) for p in prompts]
    
    results = await asyncio.gather(*tasks)
    for res in results:
        print(f"Final Result: {res}")

if __name__ == "__main__":
    asyncio.run(main())

            