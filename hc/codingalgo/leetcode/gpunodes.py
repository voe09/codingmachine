"""
Problem StatementYou are given $N$ GPU nodes. 
Some are "good" and some are "bad". 
You have access to a testing function test(S) 
which takes a subset of nodes $S$ as input:Returns False: 
If there is at least one bad node in $S$.Returns True: 
If all nodes in $S$ are good.Constraints:Parallelism: 
test(S) can be called in parallel, but a single node cannot participate in 
more than one test at the same time.Minimum Size: $|S| \geq 2$. 
You cannot test a single node by itself.Goal: Design an algorithm to 
identify all bad nodes as efficiently as possible.
"""

class GPU:

    def __init__(self, id: int, working: bool):
        self.id = id
        self.working = working

class NodeTester:

    def test(self, gpus: list[GPU]) -> bool:
        return all(gpu.working for gpu in gpus)
    
    def validate(self, gpus: list[GPU]) -> dict[int, bool]:
        # find the anchor good gpu
        n = len(gpus)
        ans = {}

        def find_anchor_idx():
            for i in range(n-1):
                for j in range(i+1, n):
                    if self.test([gpus[i], gpus[j]]):
                        return i
            return None
        
        idx = find_anchor_idx()
        if idx is None:
            raise ValueError("No anchor good GPU exists")

        # move the anchor GPU to the last
        gpus[idx], gpus[-1] = gpus[-1], gpus[idx]
        anchor = gpus[-1]        
        ans[anchor.id] = True

        def dfs(l: int, r: int):
            if r - l <= 0:
                return
            
            if r - l == 1:
                if self.test([gpus[l], anchor]):
                    ans[gpus[l].id] = True
                else:
                    ans[gpus[l].id] = False
                return
        
            if self.test(gpus[l:r]):
                for gpu in gpus[l:r]:
                    ans[gpu.id] = True
                return
            
            mid = l + (r - l) // 2
            dfs(l, mid)
            dfs(mid, r)
        
        dfs(0, n-1)
        return ans
    
def run_test(gpu_states):
    gpus = [GPU(i, working) for i, working in enumerate(gpu_states)]
    tester = NodeTester()
    return tester.validate(gpus)

gpu_states = [True, True, True, True]
print(run_test(gpu_states))

gpu_states = [True, False, True, True]
print(run_test(gpu_states))

gpu_states = [False, True, False, True, True]
print(run_test(gpu_states))

gpu_states = [True, True, False, False, True]
print(run_test(gpu_states))
