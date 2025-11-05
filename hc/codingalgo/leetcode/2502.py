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


from bisect import insort

class Allocator:

    def __init__(self, n: int):
        self.blocks = [[0, n]]
        self.allocations: dict[int, list[list[int]]] = {} # mID: [(start, size)]

    def allocate(self, size: int, mID: int) -> int:
        for i in range(len(self.blocks)):
            if self.blocks[i][1] >= size:
                start, block_size = self.blocks[i]
                self.blocks.pop(i)
                if block_size > size:
                    insort(self.blocks, [start + size, block_size - size])
                if mID not in self.allocations:
                    self.allocations[mID] = []
                
                self.allocations[mID].append([start, size])
                return start
        return -1

    def freeMemory(self, mID: int) -> int:
        if mID not in self.allocations:
            return 0

        block_size = 0
        for start, size in self.allocations[mID]:
            block_size += size
            insort(self.blocks, [start, size])
        del self.allocations[mID]

        
        merged_blocks = []
        for block in self.blocks:
            if len(merged_blocks) == 0 or merged_blocks[-1][0] + merged_blocks[-1][1] < block[0]:
                merged_blocks.append(block)
            else:
                merged_blocks[-1][1] = merged_blocks[-1][1] + block[1]
        
        self.blocks = merged_blocks
        return block_size