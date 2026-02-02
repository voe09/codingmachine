class InMemDB:

    def __init__(self):
        self.cache = [{}]

    def set(self, key: str, val: int):
        self.cache[-1][key] = val

    def get(self, key: str):
        for level in reversed(self.cache):
            if key in level:
                return level[key]
        return None

    def begin(self):
        self.cache.append({})
    
    def rollback(self):
        if len(self.cache) > 1:
            self.cache.pop()
        
    def commit(self):
        if len(self.cache) <= 1:
            return
    
        while len(self.stack) > 1:
            top_level = self.stack.pop()
            self.stack[-1].update(top_level)
