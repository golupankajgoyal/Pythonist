from heapq import *
from math import inf
# Given a list of ‘K’ sorted arrays, merge them into one sorted list.
def merge_k_list(arr):
    min_heap=[]
    merged_arr=[]
    for i in range(len(arr)):
        heappush(min_heap,(arr[i][0],0,arr[i]))
    while min_heap:
        num,index,list=heappop(min_heap)
        index+=1
        merged_arr.append(num)
        if len(list)>index:
            heappush(min_heap,(list[index],index,list))
    print(merged_arr)
# arr=[[5, 8, 9],[1, 7]]
# merge_k_list(arr)

#Given ‘M’ sorted arrays, find the K’th smallest number among all the arrays.
def find_k_smallest(arr,k):
    min_heap=[]
    for i in range(len(arr)):
        heappush(min_heap,(arr[i][0],0,arr[i]))
    count=0
    while min_heap:
        count+=1
        num,index,list=heappop(min_heap)
        if count==k:
            return num
        index+=1
        if len(list)>index:
            heappush(min_heap,(list[index],index,list))
    return -1

# arr=[[2, 6, 8], [3, 6, 7],[1, 3, 4]]
# K=5
# print(find_k_smallest(arr,K))

#Given ‘M’ sorted arrays, find the smallest range that includes at least one number from each of the ‘M’ lists.
def find_smallest_range(arr):
    min_heap=[]
    curr_max=0
    rng,range_len=(0,0),inf
    for i in range(len(arr)):
        heappush(min_heap,(arr[i][0],0,arr[i]))
        curr_max=max(curr_max,arr[i][0])

    while len(min_heap)==len(arr):
        num,index,list=heappop(min_heap)
        if range_len>(curr_max-num):
            range_len=curr_max-num
            rng=(num,curr_max)
        index+=1
        if len(list)>index:
            heappush(min_heap,(list[index],index,list))
            curr_max=max(curr_max,list[index])
    print(rng,range_len)

# arr=[[1, 9], [4, 12], [7, 10, 16]]
# find_smallest_range(arr)

# Given two sorted arrays in descending order, find ‘K’ pairs with the largest sum where
# each pair consists of numbers from both the arrays.
def k_largest_pair(arr,k):
    max_heap=[]
    count=0
    for i in range(min(k,len(arr[0]))):
        for j in range(min(k,len(arr[1]))):
            if count>k:
                curr_sum=arr[0][i]+arr[1][j]
                sum,num1,num2=heappop(max_heap)
                if curr_sum>sum:
                    heappush(max_heap,(curr_sum,arr[0][i],arr[1][j]))
                else:
                    return max_heap
            else:
                count+=1
                heappush(max_heap,(arr[0][i]+arr[1][j],arr[0][i],arr[1][j]))
    return max_heap
# arr=[[5, 2, 1], [2, -1]]
# K=3
# print(k_largest_pair(arr,K))






