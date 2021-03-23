from heapq import heappush as hpush,heappop as hpop


def equilibriumPoint(arr, N):
    # Your code here
    start = 0
    end = N - 1
    left_sum = arr[start]
    right_sum = arr[end]
    while start < end:
        if left_sum > right_sum:
            end -= 1
            right_sum += arr[end]
        elif right_sum > left_sum:
            start += 1
            left_sum += arr[start]
        else:
            start+=1
            end-=1
            left_sum += arr[start]
            right_sum += arr[end]
    if left_sum == right_sum and start==end:
        return start + 1
    else:
        return -1
# arr=[4,42,27,16,28,3,4,5,9,3,31,5,5,29,10,18,35,35,33,19,41,23,8,32,9,5,8,18,35,13,6,7,6,10,11,13,37,2,25,7,28,43]
# arr=[26,26]
# n=len(arr)
# print( equilibriumPoint(arr, n))

def find_pair(arr,num,n):
    start=0
    end=0
    while end<n:
        diff=arr[end]-arr[start]
        if diff>num:
            start+=1
        elif diff<num:
            end+=1
        else:
            return 1
        print(diff,start,end)
    return -1
# num=78
# arr=[5, 20, 3, 2, 50, 80]
# arr=sorted(arr)
# n=len(arr)
# print(find_pair(arr,num,len(arr)))

#Convert array into Zig-Zag fashion
# Given an array of DISTINCT elements, rearrange the elements of array in zig-zag fashion in O(n) time.
# The converted array should be in form a < b > c < d > e < f.
def zig_zag_conversion(arr,n):
    flag=0
    for i in range(n-1):
        if flag==0 and arr[i+1]<arr[i]:
            arr[i],arr[i+1]=arr[i+1],arr[i]
        elif flag==1 and arr[i+1]>arr[i]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
        flag=1-flag
    print(arr)
# arr=[1, 4, 3, 2]
# zig_zag_conversion(arr,len(arr))

# Minimum Platforms
# Given arrival and departure times of all trains that reach a railway station,
# the task is to find the minimum number of platforms required for the railway station so that no train waits.
# We are given two arrays which represent arrival and departure times of trains that stop.
def min_stations(arr,dep):
    arr_time = []
    n=len(arr)
    # sort the arrival time
    for i in range(n):
        hpush(arr_time, (arr[i], i))
    min_heap = []
    ans = 0
    for i in range( len(arr)):
        curr_arr, index = hpop(arr_time)
        while min_heap and curr_arr >min_heap[0]:
            hpop(min_heap)
        hpush(min_heap, dep[index])
        ans = max(ans, len(min_heap))
    return ans

# arr=[900, 940, 950, 1100, 1500, 1800]
# dep=[910, 1200, 1120, 1130, 1900, 2000]
# print(min_stations(arr,dep))















