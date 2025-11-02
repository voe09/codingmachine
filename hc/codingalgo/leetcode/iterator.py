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








from abc import ABC, abstractmethod


class IteratorInterface(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass


class ListIterator(IteratorInterface):

    def __init__(self, data: list[int]):
        self.index = 0
        self.data = data

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration("No more element")
        self.index += 1
        return self.data[self.index - 1]

    def get_state(self):
        return self.index

    def set_state(self, state):
        self.index = state 
        


iterator = ListIterator((1, 2, 3))
for i in range(2):
    print(next(iter(iterator)))

state = iterator.get_state()
new_iterator = ListIterator((1, 2, 3))
new_iterator.set_state(state)
print(next(iter(new_iterator)))

try:
    print(next(iter(new_iterator)))
except StopIteration:
    print("no more data")


class MultiListIterator(IteratorInterface):

    def __init__(self, iters: list[ListIterator]):
        self.index = 0
        self.iters = iters

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.iters):
            it = self.iters[self.index]
            try:
                value = next(iter(it))
                return value
            except StopIteration:
                self.index += 1
        raise StopIteration("no more data")

    def get_state(self):
        if self.index < len(self.iters):
            return (self.index, self.iters[self.index].get_state())
        
        else:
            return (self.index, -1)

    def set_state(self, state):
        self.index = state[0]
        if self.index < len(self.iters):
            self.iters[self.index].set_state(state[1])

iterator = MultiListIterator([
    ListIterator((1, 2, 3)),
    ListIterator((4, 5)),
    ListIterator(()),
    ListIterator((6, 7)),
])

for i in range(5):
    print(next(iter(iterator)))

state = iterator.get_state()

new_iterator = MultiListIterator([
    ListIterator((1, 2, 3)),
    ListIterator((4, 5)),
    ListIterator(()),
    ListIterator((6, 7)),
])

new_iterator.set_state(state)
for i in range(2):
    print(next(iter(new_iterator)))

try:
    next(iter(new_iterator))
except StopIteration:
    print("no more data")


from abc import ABC, abstractmethod
from collections.abc import Iterator


class IteratorInterface(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass


class ListIterator(IteratorInterface):

    def __init__(self, data: list[int]):
        self.index = 0
        self.data = data

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration("No more element")
        self.index += 1
        return self.data[self.index - 1]

    def get_state(self):
        return self.index

    def set_state(self, state):
        self.index = state 
        


iterator = ListIterator((1, 2, 3))
for i in range(2):
    print(next(iter(iterator)))

state = iterator.get_state()
new_iterator = ListIterator((1, 2, 3))
new_iterator.set_state(state)
print(next(iter(new_iterator)))

try:
    print(next(iter(new_iterator)))
except StopIteration:
    print("no more data")


class MultiListIterator(IteratorInterface):

    def __init__(self, iters: list[ListIterator]):
        self.index = 0
        self.iters = iters

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.iters):
            it = self.iters[self.index]
            try:
                value = next(iter(it))
                return value
            except StopIteration:
                self.index += 1
        raise StopIteration("no more data")

    def get_state(self):
        if self.index < len(self.iters):
            return (self.index, self.iters[self.index].get_state())
        
        else:
            return (self.index, -1)

    def set_state(self, state):
        self.index = state[0]
        if self.index < len(self.iters):
            self.iters[self.index].set_state(state[1])

iterator = MultiListIterator([
    ListIterator((1, 2, 3)),
    ListIterator((4, 5)),
    ListIterator(()),
    ListIterator((6, 7)),
])

for i in range(5):
    print(next(iter(iterator)))

state = iterator.get_state()

new_iterator = MultiListIterator([
    ListIterator((1, 2, 3)),
    ListIterator((4, 5)),
    ListIterator(()),
    ListIterator((6, 7)),
])

new_iterator.set_state(state)
for i in range(2):
    print(next(iter(new_iterator)))

try:
    next(iter(new_iterator))
except StopIteration:
    print("no more data")


class JsonFileIterator(Iterator):

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'r')
        self.offset = self.offset
    
    def __iter__(self):
        return self

    def __next__(self):
        self.file.seek(self.offset)
        line = self.file.readline()
        if not line:
            raise StopIteration
        self.offset = self.file.tell()

    def get_state(self):
        return self.offset
    
    def set_state(self, state):
        self.offset = state

class MultiJsonFileIterator(Iterator):

    def __init__(self, json_files: list[str]):
        self.index = 0
        self.json_iterators = [JsonFileIterator(f) for f in json_files]

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.json_iterators):
            it = self.json_iterators[self.index]
            try:
                line = next(iter(it))
                return line
            except:
                self.index += 1
        raise StopIteration

    def get_state(self):
        if self.index < len(self.json_iterators):
            return (self.index, self.json_iterators[self.index].get_state())
    
        return (self.index, -1)

    def set_state(self, state):
        self.index = state[0]
        if self.index < len(self.json_iterators):
            self.json_iterators[self.index].set_state(state[1])
