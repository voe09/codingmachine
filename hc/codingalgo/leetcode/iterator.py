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







import unittest

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeVar, Generic, Iterator, Any

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

T = TypeVar("T")

class ResumableIterator(IteratorInterface, Generic[T]):
    """Templatized iterator supporting suspension/resumption over any iterable."""

    def __init__(self, iterable: Iterable[T]):
        self.source = iter(iterable)
        self.pos = 0
        self.stop = False

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        # If we've already seen this item, return from buffer
        if self.stop:
            raise StopIteration
        try:
            value = next(self.source)
            self.pos += 1
            return value
        except StopIteration as err:
            self.stop = True
            raise err

    def get_state(self):
        """Return a serializable snapshot of iteration progress."""
        return (self.pos, self.stop)

    def set_state(self, state):
        """Restore iteration progress."""
        self.pos = state[0]
        self.stop = state[1]
        if not self.stop:
            for _ in range(self.pos):
                next(self.source)



class MultipleResumableIterator(IteratorInterface, Generic[T]):

    def __init__(self, iterator: Iterable[ResumableIterator[T]]):
        self.sources = iterator
        self.iter = None
        self.pos = 0
        self.stop = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration
        
        while True:
            if self.iter is None:
                try:
                    self.iter = next(self.sources)
                    self.pos += 1
                except StopIteration:
                    self.stop = True
                    raise StopIteration

            try:
                value = next(self.iter)
                return value
            except StopIteration:
                self.iter = None
    
    def get_state(self):
        if self.stop:
            return {
                "stop": self.stop,
                "pos": self.pos,
            }
        else:
            return {
                "stop": self.stop,
                "pos": self.pos,
                "iter": None if self.iter is None else self.iter.get_state(),
            }

    def set_state(self, state):
        stop = state["stop"]
        if stop:
            self.stop = stop
            self.pos = state["pos"]
        else:
            self.stop = stop
            self.pos = state["pos"]
            iterator = None
            for _ in range(self.pos):
                iterator = next(self.sources)
            self.iter = iterator
            if iterator is not None:
                self.iter.set_state(state["iter"])


class TestIterator(unittest.TestCase):

    def setUp(self):
        def data_stream():
            for x in range(5):
                print(f"Producing {x}")
                yield x
        self.data = data_stream

    def test_iterator_loop(self):
        iterator = ResumableIterator(self.data())
        for i in range(5):
            next_value = next(iter(iterator))
            self.assertEqual([0, 1, 2, 3, 4][i], next_value)

        with self.assertRaises(StopIteration):
            next(iter(iterator))
        
        self.assertTrue(iterator.stop)

    def test_get_state_and_set_state(self):
        states = []
        expected_value = []
        iterator = ResumableIterator(self.data())
        for i in range(5):
            state = iterator.get_state()
            states.append(state)
            expected_value.append(next(iter(iterator)))
        
        with self.assertRaises(StopIteration):
            next(iter(iterator))
        
        states.append(iterator.get_state())

        values = []
        for state in states[:-1]:
            new_iterator = ResumableIterator(self.data())
            new_iterator.set_state(state)
            values.append(next(iter(new_iterator)))
        self.assertEqual(expected_value, values)

        new_iterator = ResumableIterator(self.data())
        new_iterator.set_state(states[-1])
        with self.assertRaises(StopIteration):
            next(iter(new_iterator))
        

class TestMultipleResumableIterator(unittest.TestCase):

    def setUp(self):
        # Simulate multiple "JSON files" as lists
        self.file_data = [
            [1, 2, 3],       # File 1
            ["a", "b"],      # File 2
            [True, False]    # File 3
        ]
        # Wrap each list in a ResumableIterator
        self.iterators = [ResumableIterator(data) for data in self.file_data]

    def test_iterator_loop(self):
        # Create MultipleResumableIterator
        multi_iter = MultipleResumableIterator(iter(self.iterators))
        
        # Expected flattened sequence
        expected = [1, 2, 3, "a", "b", True, False]
        output = []
        for item in multi_iter:
            output.append(item)
        
        self.assertEqual(output, expected)
        self.assertTrue(multi_iter.stop)

    def test_get_state_and_set_state(self):
        # Create MultipleResumableIterator
        multi_iter = MultipleResumableIterator(iter(self.iterators))
        states = []
        values = []

        # Consume items one by one, save state before each
        for _ in range(3):
            state = multi_iter.get_state()
            states.append(state)
            values.append(next(multi_iter))  # consume item

        # Save state in the middle
        mid_state = multi_iter.get_state()

        # Consume rest
        while True:
            try:
                values.append(next(multi_iter))
            except StopIteration:
                break

        # Test resuming from intermediate states
        for idx, state in enumerate(states):
            # Create fresh iterators for resumption
            iterators_copy = [ResumableIterator(data) for data in self.file_data]
            multi_iter_resume = MultipleResumableIterator(iter(iterators_copy))
            multi_iter_resume.set_state(state)
            resumed_value = next(multi_iter_resume)
            self.assertEqual(resumed_value, values[idx])

        # Test resuming from "end" state
        iterators_copy = [ResumableIterator(data) for data in self.file_data]
        multi_iter_resume = MultipleResumableIterator(iter(iterators_copy))
        multi_iter_resume.set_state(mid_state)
        resumed_values = [next(multi_iter_resume) for _ in range(len(values) - 3)]
        self.assertEqual(resumed_values, values[3:])

        # Ensure StopIteration is raised at the end
        with self.assertRaises(StopIteration):
            next(multi_iter_resume)


unittest.main(verbosity=2)