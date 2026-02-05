from collections import deque

class MockLLM:
    def generate(self, batch_prefix: list[str]) -> list[str]:
        # Simulating distinct token generation for each request in the batch
        return [f"t{i+1}_" for i in range(len(batch_prefix))]

class Request:
    def __init__(self, request_id: str, prefix: str, max_token: int, stop_token: str):
        self.request_id = request_id
        self.res = prefix
        self.max_token = max_token
        self.stop_token = stop_token
        self.finish = False
        self.generated_count = 0 # Track tokens, not characters

    def add_token(self, token: str):
        self.res += token
        self.generated_count += 1
        # Termination condition logic
        if self.generated_count >= self.max_token or token == self.stop_token:
            self.finish = True

class InferenceEngine:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.tasks = deque([])
        self.slots = [None] * batch_size
        self.availables = deque(list(range(batch_size)))
        self.occupied_slots = {} # request_id -> slot_idx for faster lookup
        self.model = MockLLM()
    
    def add_request(self, request: Request):
        self.tasks.append(request)

    def _fill_slots(self):
        while self.tasks and self.availables:
            req = self.tasks.popleft()
            slot_idx = self.availables.popleft()
            self.slots[slot_idx] = req
            self.occupied_slots[req.request_id] = slot_idx
    
    def run(self):
        # Continue if there are tasks waiting OR slots are being worked on
        while self.tasks or len(self.availables) < self.batch_size:
            self._fill_slots()

            # Identify which slots currently have an active request
            active_indices = [i for i, req in enumerate(self.slots) if req is not None]
            if not active_indices:
                break

            # Step 2: Iterative Decoding
            batch_inputs = [self.slots[i].res for i in active_indices]
            next_tokens = self.model.generate(batch_inputs)

            # Step 3: Update and Recycle
            for i, next_token in enumerate(next_tokens):
                slot_idx = active_indices[i]
                req = self.slots[slot_idx]
                
                req.add_token(next_token)
                
                if req.finish:
                    print(f"Finished: {req.request_id} | Result: {req.res}")
                    # Free resources
                    self.slots[slot_idx] = None
                    self.availables.append(slot_idx)
                    del self.occupied_slots[req.request_id]

# --- Test Case ---
engine = InferenceEngine(batch_size=2)
engine.add_request(Request("REQ-A", "Start", max_token=5, stop_token="STOP"))
engine.add_request(Request("REQ-B", "Begin", max_token=2, stop_token="STOP"))
engine.add_request(Request("REQ-C", "Go", max_token=3, stop_token="STOP"))

engine.run()