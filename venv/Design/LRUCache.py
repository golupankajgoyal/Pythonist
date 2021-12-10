from collections import deque, defaultdict, Counter


class Node:
    def __init__(self, value):
        self.val = value
        self.prev = None
        self.next = None


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.head = Node("#")
        self.tail = Node("#")
        self.size = 0
        self.dic = defaultdict(Node)
        self.connect(self.head, self.tail)

    def connect(self, node1, node2):
        node1.next = node2
        node2.prev = node1

    def append(self, node):
        self.connect(self.tail.prev, node)
        self.connect(node, self.tail)
        self.dic[node.val] = node
        self.size += 1
        if self.size > self.capacity:
            self.delete(self.head.next)
            self.size -= 1

    def delete(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p
        del self.dic[node.val]

    def get(self, key):
        if key in self.dic:
            node = self.dic[key]
            self.delete(node)
            self.append(node)
            return self.dic[key].val
        return -1

    def put(self, key, val):
        if key in self.dic:
            self.delete(self.dic[key])
        node = Node(val)
        self.dic[key] = node
        self.append(node)
