class Allocator:

    def __init__(self, n: int):
        self.allocation = {} # mID: [(start_index, block_size)]
        self.heap = [[0, n]] # min heap, (start_idx, block_size)

    def allocate(self, size: int, mID: int) -> int:
        for i in range(len(self.heap)):
            block = self.heap[i]
            if block[1] >= size:
                self.heap.pop(i)
                if mID not in self.allocation:
                    self.allocation[mID] = []
                self.allocation[mID].append([block[0], size])
                if block[1] > size:
                    self.heap.insert(i, [block[0] + size, block[1] - size])
                return block[0]
        return -1                
        
    def freeMemory(self, mID: int) -> int:
        if mID not in self.allocation:
            return 0
        
        free = 0
        for block in self.allocation[mID]:
            insort(self.heap, block)
            free += block[1]
    
        del self.allocation[mID]
        
        merge = []
        for block in self.heap:
            if not merge or (merge[-1][0] + merge[-1][1]) < block[0]:
                merge.append(block)
            else:
                merge[-1][1] = merge[-1][1] + block[1]
    
        self.heap = merge
        return free

        


# Your Allocator object will be instantiated and called as such:
# obj = Allocator(n)
# param_1 = obj.allocate(size,mID)
# param_2 = obj.freeMemory(mID)