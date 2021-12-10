from math import inf, ceil, floor
import heapq
import bisect
from collections import deque, defaultdict, Counter
import sys
# Used to compare in builtIn sorted function
from functools import cmp_to_key

def rotateRight(head, k):
    if head is None: return head
    size = 0
    end, curr = None, head
    while curr:
        end = curr
        curr = curr.next
        size += 1
    k, curr, prev = size - k % size, head, None
    while k > 0:
        prev = curr
        curr = curr.next
        k -= 1
    if curr:
        end.next = head
        prev.next = None
    return curr or head




























































