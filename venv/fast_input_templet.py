import os
import sys
# For flush output
# sys.stdout.flush()
from collections import deque, defaultdict, Counter, OrderedDict
from math import ceil, sqrt, hypot, factorial, pi, sin, cos, radians,inf
from heapq import heappush, heappop, heapify, nlargest, nsmallest
from itertools import permutations
sys.setrecursionlimit(1 << 30)
from io import BytesIO, IOBase
import bisect
import random

ins = lambda: [int(x) for x in input()]
inp = lambda: int(input())
inps = lambda: [int(x) for x in input().split()]
from fractions import Fraction as F

md = pow(10, 9) + 7

N = 2 * (10 ** 5) + 7

dp = [[-1] * N for _ in range(int(math.log2(N)) + 1)]

def dice_combination_dp(num):
    dp=[-1 for i in range(num+1)]
    dp[0]=1
    for  i in range(1,num+1):
        temp=0
        for j in range(1,7):
            if i-j>=0:
                temp+=dp[i-j]
        dp[i]=temp
    return dp[num]

# dp2=[1]*(1000001)
def main():
    num = inp()
    print(dice_combination_dp(num) % md)


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