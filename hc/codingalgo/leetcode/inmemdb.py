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



class DB:

    def __init__(self):
        self.data = {}

    def set(self, key: str, field: str, value: str):
        if key not in self.data:
            self.data[key] = {}
        
        self.data[key][field] = value
        return

    def get(self, key: str, field: str):
        if key not in self.data:
            return ""
        if field not in self.data[key]:
            return ""
        return self.data[key][field]
    
    def delete(self, key: str, field: str):
        if key not in self.data or field not in self.data[key]:
            return False
        
        self.data[key].pop(field)
        if len(self.data[key]) == 0:
            self.data.pop(key)
        return True
    
