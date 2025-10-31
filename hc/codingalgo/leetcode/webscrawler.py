class URL:
    def __init__(self, addr: str):
        self.addr = addr
        self.children = []

    def add_child(self, url: "URL"):
        self.children.append(url)

    def __repr__(self) -> str:
        return self.addr

    def __eq__(self, other):
        if not isinstance(other, URL):
            return False
        return self.addr == other.addr

    def __hash__(self):
        return hash(self.addr)

class WebCrawler:

    def __init__(self):
        self.visited = set()

    def craw(self, url: URL):
        if url in self.visited:
            return
        
        queue = [url]
        while len(queue) > 0:
            node = queue.pop(0)
            self.visited.add(node)
            for child in node.children:
                if child not in self.visited:
                    queue.append(child)


url1 = URL("link1")
url2 = URL("link2")
url3 = URL("link3")
url1.add_child(url2)
url1.add_child(url3)

crawler = WebCrawler()
crawler.craw(url1)

for link in crawler.visited:
    print(link)


import threading

class URL:
    def __init__(self, addr: str):
        self.addr = addr
        self.children = []

    def add_child(self, url: "URL"):
        self.children.append(url)

    def __repr__(self) -> str:
        return self.addr

    def __eq__(self, other):
        if not isinstance(other, URL):
            return False
        return self.addr == other.addr

    def __hash__(self):
        return hash(self.addr)

class MultithreadingWebCrawler:

    def __init__(self):
        self.visited = set()
        self.lock = threading.Lock()

    def craw(self, url: URL):
        with self.lock:
            if url in self.visited:
                return
            self.visited.add(url)

        threads = []
        for child in url.children:
            with self.lock:
                if child not in self.visited:
                    t = threading.Thread(target=self.craw, args=(child,))
                    t.start()
                    threads.append(t)
        
        for t in threads:
            t.join()



url1 = URL("link1")
url2 = URL("link2")
url3 = URL("link3")
url1.add_child(url2)
url1.add_child(url3)

crawler = MultithreadingWebCrawler()
crawler.craw(url1)

for link in crawler.visited:
    print(link)



from concurrent.futures import ThreadPoolExecutor, wait
import threading

class URL:
    def __init__(self, addr: str):
        self.addr = addr
        self.children = []

    def add_child(self, url: "URL"):
        self.children.append(url)

    def __repr__(self) -> str:
        return self.addr

    def __eq__(self, other):
        if not isinstance(other, URL):
            return False
        return self.addr == other.addr

    def __hash__(self):
        return hash(self.addr)

class MultithreadingWebCrawler:

    def __init__(self, max_workers: int=4):
        self.visited = set()
        self.lock = threading.Lock()
        self.workers = ThreadPoolExecutor(max_workers=max_workers)

    def craw(self, url: URL):
        with self.lock:
            if url in self.visited:
                return
            self.visited.add(url)

        futures = []
        for child in url.children:
            with self.lock:
                if child not in self.visited:
                    future = self.workers.submit(self.craw, child)
                    futures.append(future)
        
        while futures:
            done, _  = wait(futures)
            futures = [f for f in futures if not f.done()]


url1 = URL("link1")
url2 = URL("link2")
url3 = URL("link3")
url1.add_child(url2)
url1.add_child(url3)

crawler = MultithreadingWebCrawler()
crawler.craw(url1)

for link in crawler.visited:
    print(link)