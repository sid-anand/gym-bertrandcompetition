class FixedList(list):

    def __init__(self, n):
        self.n = n

    def add(self, item):
        list.insert(self, 0, item)
        if len(self) > self.n: del self[-1]