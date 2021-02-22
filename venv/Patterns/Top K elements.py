from heapq import *
from collections import defaultdict, deque


# Given an unsorted array of numbers, find the ‘K’ largest numbers in it.
def find_k_large_num(arr, k):
    minheap = []
    for num in arr[:k]:
        heappush(minheap, num)
    for num in arr[k:]:
        if num > minheap[0]:
            heappop(minheap)
            heappush(minheap, num)
    print(minheap)


# arr=[5, 11, 12, -1, 11]
# K = 3
# find_k_large_num(arr,K)
# Given an array of points in the a 2D2D plane, find ‘K’ closest points to the origin.
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        return self.calc_dist() > other.calc_dist()

    def calc_dist(self):
        return self.x ** 2 + self.y ** 2


def find_k_nearest(arr, k):
    points = []
    for point in arr:
        new_point = Point(point[0], point[1])
        points.append(new_point)
    max_heap = []
    for i in points[0:k]:
        heappush(max_heap, i)

    for i in points[k:]:
        if max_heap[0].calc_dist() > i.calc_dist():
            heappop(max_heap)
            heappush(max_heap, i)
    for point in max_heap:
        print(point.x, point.y)


# point = [[1, 3], [3, 4], [2, -1]]
# K = 2
# find_k_nearest(point,2)

# Given ‘N’ ropes with different lengths, we need to connect these ropes into one big rope with minimum cost.
# The cost of connecting two ropes is equal to the sum of their lengths.

def connect_rope(arr):
    cost = 0
    heapify(arr)
    while len(arr) > 1:
        curr_cost = heappop(arr) + heappop(arr)
        cost += curr_cost
        heappush(arr, curr_cost)
    print(cost)


# arr=[1, 3, 11, 5, 2]
# connect_rope(arr)

# Given an unsorted array of numbers, find the top ‘K’ frequently occurring numbers in it.

def find_k_frequent(arr, k):
    frq = defaultdict()
    for num in arr:
        if num in frq:
            frq[num] += 1
        else:
            frq[num] = 1

    min_heap = []
    for num in arr[:k]:
        heappush(min_heap, num)
    for num in arr[k:]:
        if frq[min_heap[0]] < frq[num]:
            heapreplace(min_heap, num)
    print(min_heap)


# arr=[5, 12, 11, 3, 11]
# K = 2
# find_k_frequent(arr,K)

# Given a string, sort it based on the decreasing frequency of its characters.

def arrange_by_frq(string):
    map = defaultdict()
    for char in string:
        map[char] = map.get(char, 0) + 1
    max_heap = []
    for char, frq in map.items():
        heappush(max_heap, (-frq, char))
    result = ""
    while len(max_heap) > 0:
        frq, char = heappop(max_heap)
        for i in range(-frq):
            result += char
    print(result)


# string="abcbab"
# arrange_by_frq(string)


# Given a sorted number array and two integers ‘K’ and ‘X’, find ‘K’ closest numbers to ‘X’ in the array.
# Return the numbers in the sorted order. ‘X’ is not necessarily present in the array.

def find_k_nearestOfX(arr, k, x):
    nearest = find_nearest(arr, x)
    print(arr[nearest])
    left = nearest - 1
    right = nearest + 1
    result = deque()
    result.append(arr[nearest])
    k -= 1
    while k > 0:
        if left >= 0 and right < len(arr):
            if abs(x - arr[left]) > abs(x - arr[right]):
                result.append(arr[right])
                right += 1
            else:
                result.appendleft(arr[left])
                left -= 1
        elif left >= 0:
            result.appendleft(arr[left])
            left -= 1
        elif right < len(arr):
            result.append(arr[right])
            right += 1
        k -= 1
    print(result)


def find_nearest(arr, x):
    start, end = 0, len(arr) - 1
    while start < end:
        mid = start + (end - start) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            start = mid + 1
        else:
            end = mid - 1
    return start


# arr=[2, 4, 5, 6, 9]
# K = 3
# X = 10
# find_k_nearestOfX(arr,K,X)

# Given a string, find if its letters can be rearranged in such a way that
# no two same characters come next to each other.
def rearrange_string(string):
    map = defaultdict()
    for char in string:
        map[char] = map.get(char, 0) + 1
    max_heap = []
    for char, freq in map.items():
        heappush(max_heap, (-freq, char))

    prev_char, prev_freq = None, 0
    result = ""
    while max_heap:
        freq, char = heappop(max_heap)
        freq = -freq
        result = result + char
        freq -= 1
        if prev_char and prev_freq > 0:
            heappush(max_heap, (-prev_freq, prev_char))
        prev_char = char
        prev_freq = freq
    return result if len(result) == len(string) else ""


# str="Programming"
# print(rearrange_string(str))

# Given a string and a number ‘K’, find if the string can be rearranged
# such that the same characters are at least ‘K’ distance apart from each other.
def rearrange_k_distance_apart(string, k):
    if k <= 1:
        return ""
    map = defaultdict()
    for char in string:
        map[char] = map.get(char, 0) + 1
    max_heap = []
    for char, freq in map.items():
        heappush(max_heap, (-freq, char))
    queue = deque()
    result = ""
    while max_heap:
        freq, char = heappop(max_heap)
        freq = -freq
        result = result + char
        freq -= 1
        if len(queue) == k - 1:
            prev_freq, prev_char = queue.popleft()
            if prev_freq > 0:
                heappush(max_heap, (-prev_freq, prev_char))
        queue.append((freq, char))
    print(queue)
    return result if len(result) == len(string) else ""


# str="aaabcddd"
# K=4
# print(rearrange_k_distance_apart(str,K))

# You are given a list of tasks that need to be run, in any order, on a server.
# Each task will take one CPU interval to execute but once a task has finished,
# it has a cooling period during which it can’t be run again. If the cooling period for all tasks
# is ‘K’ intervals, find the minimum number of CPU intervals that the server needs to finish all tasks.
# If at any time the server can’t execute any task then it must stay idle.

def scheduling_task(arr, k):
    map = defaultdict()
    for char in arr:
        map[char] = map.get(char, 0) + 1
    max_heap = []
    for char, freq in map.items():
        heappush(max_heap, (-freq, char))
    result = 0
    while max_heap:
        waiting_task = []
        n = k + 1
        while n > 0 and max_heap:
            result += 1
            freq, char = heappop(max_heap)
            freq = -freq
            freq -= 1
            if freq > 0:
                waiting_task.append((-freq, char))
            n -= 1
        for task in waiting_task:
            heappush(max_heap, task)
        if max_heap:
            result += n
    return result
arr=["a", "b","a"]
K=3
print(scheduling_task(arr,K))

















