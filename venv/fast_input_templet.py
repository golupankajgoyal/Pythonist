import os
import sys

sys.setrecursionlimit(1 << 30)
from io import BytesIO, IOBase
from collections import defaultdict,deque
import math
import bisect
from math import inf
import random

ins = lambda: [int(x) for x in input()]
inp = lambda: int(input())
inps = lambda: [int(x) for x in input().split()]
from fractions import Fraction as F

md = pow(10, 9) + 7

N = 2 * (10 ** 5) + 7

dp = [[-1] * N for _ in range(int(math.log2(N)) + 1)]
def binary_lift(node,parent,tree,dp):
    dp[node]=[-1]*20
    dp[node][0]=parent
    for i in range(1,20):
        if dp[dp[node][i-1]]!=[]:
            dp[node][i]=dp[dp[node][i-1]][i-1]
        else:
            break
    for child in tree[node]:
        if child !=parent:
            binary_lift(child,node,tree,dp)

def bit_set_stack(pos,num,stack):
    if num==0:
        return
    pos=0
    while num>0:
        if (num&1)!=0:
            stack.append(pos)
        pos+=1
        num>>=1
    return stack

def find_kth_parent(node,dp,stack):
    temp=node
    while stack:
        power=stack.pop()
        temp=dp[temp][power]
    return temp



# dp2=[1]*(1000001)
def main():
    dp = [[-1] * N for _ in range(int(math.log2(N)) + 1)]
    tree = defaultdict(list)
    n, q = inps()
    parents = inps()
    for i in range(len(parents)):
        tree[parents[i]].append(i + 2)
        tree[i + 2].append(parents[i])
    binary_lift(1, -1, tree, dp)
    for i in range(q):
        node, k = inps()
        stack = deque()
        bit_set_stack(0, k, stack)
        print(find_kth_parent(node, dp, stack))


    # n, q = inps()
    # a = inps()
    # for i in range(2, n + 1):
    #     b = a[i - 2]
    #     dp[0][i] = b
    #
    # for i in range(1, int(math.log2(N)) + 1):
    #     for j in range(2, n + 1):
    #         dp[i][j] = dp[i - 1][dp[i - 1][j]]
    #
    # for i in range(q):
    #     emp, k = inps()
    #     b = emp
    #     m = 0
    #     while k:
    #         if k & 1:
    #             b = dp[m][b]
    #             # m=0
    #         k >>= 1
    #         m += 1
    #     print(b)


BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

# endregion

if __name__ == "__main__":
    main()