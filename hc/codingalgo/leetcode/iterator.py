from abc import ABC, abstractmethod

class ResumableIterator(ABC):

    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass


class ResumableListIterator(ResumableIterator):

    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        res = self.data[self.index]
        self.index += 1
        return res

    def get_state(self) -> int:
        return self.index

    def set_state(self, state: int):
        if not isinstance(state, int):
            raise ValueError(f"state {state} must be a integer")
        
        if not (0 <= state <= len(self.data)):
            raise ValueError(f"state {state} must be an integer between 0 and {len(self.data)}")

        self.index = state

def test_resumable_list_iterator():
    data = (1, 2, 3, 4, 5, 6)
    iterator = ResumableListIterator(data)

    # success
    for i in iterator:
        print(i)
    
    try:
        next(iterator)
    except StopIteration:
        print("stop interation")
    
    new_iterator = ResumableListIterator(data)
    next(new_iterator)
    next(new_iterator)
    state = new_iterator.get_state()
    print(state)
    new_iterator = ResumableListIterator(data)
    new_iterator.set_state(state)
    print(new_iterator.index)

# test_resumable_list_iterator()


class ResumableMultiFileIterator(ResumableIterator):

    def __init__(self, iterators: list[ResumableIterator]):
        self.iterators = iterators
        self.index = 0

    def __next__(self) -> int:
        while self.index < len(self.iterators):
            iterator = self.iterators[self.index]
            try:
                return next(iterator)
            except StopIteration:
                self.index += 1
        
        raise StopIteration

    def get_state(self) -> tuple[int, int]:
        if self.index >= len(self.iterators):
            return (self.index, -1)
        else:
            return (self.index, self.iterators[self.index].get_state())

    def set_state(self, state: tuple[int, int]):
        idx, iter_idx = state
        self.index = idx
        if idx < len(self.iterators):
            self.iterators[idx].set_state(iter_idx)


def test_resumable_multi_list_iterator():
    iterator1 = ResumableListIterator((1, 2, 3))
    iterator2 = ResumableListIterator((4, 5))
    iterator3 = ResumableListIterator((6, 7, 8))
    iterator = ResumableMultiFileIterator((iterator1, iterator2, iterator3))


    # success
    for i in iterator:
        print(i)
    
    try:
        next(iterator)
    except StopIteration:
        print("stop interation")
    

    iterator1 = ResumableListIterator((1, 2, 3))
    iterator2 = ResumableListIterator((4, 5))
    iterator3 = ResumableListIterator((6, 7, 8))
    iterator = ResumableMultiFileIterator((iterator1, iterator2, iterator3))

    for i in range(4):
        next(iterator)
    state = iterator.get_state()
    print(state)

    iterator = ResumableMultiFileIterator((iterator1, iterator2, iterator3))
    print(iterator.get_state())

    iterator.set_state(state)
    print(iterator.get_state())
    # new_iterator = ResumableListIterator(data)
    # next(new_iterator)
    # next(new_iterator)
    # state = new_iterator.get_state()
    # print(state)
    # new_iterator = ResumableListIterator(data)
    # new_iterator.set_state(state)
    # print(new_iterator.index)

test_resumable_multi_list_iterator()