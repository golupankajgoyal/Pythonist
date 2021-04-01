from collections import deque, defaultdict, Counter, OrderedDict
from math import ceil, sqrt, hypot, factorial, pi, sin, cos, radians, log2
from heapq import heappush, heappop, heapify, nlargest, nsmallest
import os
import sys

# find the number in a given range R that has sum equals to x
def count_numbers(n,sum,tight,dp,r):
    if sum<0:
        return 0
    if  n==1:
        if 0<=sum<=9:
            return 1
        else:
            return 0
    if dp[n][sum][tight]!=-1:
        return dp[n][sum][tight]
    up_bound= 9 if tight==0 else int(r[len(r)-n])
    count=0
    for  i in range(up_bound+1):
        count+=count_numbers(n-1,sum-i,tight&(i==up_bound),dp,r)
    dp[n][sum][tight]=count
    return count

# r=11235
# r=str(r)
# n=len(r)
# tight=1
# sum=5
# dp=[[[-1 for i in range(2)]for j in range(sum+1)]for  k in range(n+1)]
# print(count_numbers(n,sum,tight,dp,r))

# Boaring Number

def count_boring_num(n,status,tight,r,dp):
    if n==0:
        return 1
    ans=0
    up_bound=9 if tight!=1 else int(r[len(r)-n])
    for i in range(status,up_bound+1,2):
        ans+=count_boring_num(n-1,1-status,tight&(i==up_bound),r,dp)
    return ans

# num="15"
# n=len(num)
# tight=1
# status=1
# dp=[]
# print(count_boring_num(n,status,tight,num,dp))
# temp=0

# Digit Sum
# For a pair of integers a and b, the digit sum of the interval [a,b] is defined as the sum of all
# digits occurring in all numbers between (and including) a and b. For example,
# the digit sum of [28, 31] can be calculated as:
def count_numbers(n,tight,r,dp):
    if tight==0:
        dp[n][tight]=10**(n)
        return dp[n][tight]
    if n==0 and tight==1:
        dp[n][tight] = 1
        return 1
    if dp[n][tight]!=1:
        return d[n][tight]
    ans=0
    ub=9 if tight==0 else int(r[len(r)-n])
    for i in range(ub+1):
        ans+=count_numbers(n-1,tight&(i==ub),r,dp)
    dp[n][tight]=ans
    return ans

def digit_sum(n,tight,r,dp,count):
    if n==0:
        return 0
    if dp[n][tight]!=-1:
        return dp[n][tight]
    ans=0
    ub=9 if tight==0 else int(r[len(r)-n])
    for i in range(ub+1):
        ans+=(i*count[n-1][tight&(i==ub)])
        ans+=digit_sum(n-1,tight&(i==ub),r,dp,count)
    dp[n][tight]=ans
    return ans

# test case
# 3
# 0 10
# 28 31
# 1234 56789

# t=int(input())
# for _ in range(t):
#     r=list(map(int,input().split()))
#     r[0]=max(r[0]-1,0)
#     ans=0
#     for k in range(2):
#         n=str(r[k])
#         size=len(n)
#         count=[[1 for  i in range(2)]for j in range(size+1)]
#         count_numbers(size,1,n,count)
#         dp=[[-1 for i in range(2)]for j in range(size+1)]
#         ans=digit_sum(size,1,n,dp,count)-ans
#     print(ans)

# Counting Numbers
def counting_numbers(n,prev,tight,trail,r,dp):
    if n==0:
        return 1
    if dp[n][prev][trail][tight]!=-1:
        return dp[n][prev][trail][tight]
    ub=9 if tight==0 else int(r[len(r)-n])
    ans=0
    for i in range(ub+1):
        if trail==1 or i!=prev:
            ans+=counting_numbers(n-1,i,tight&(i==ub),trail&(i==0),r,dp)
    dp[n][prev][trail][tight]=ans
    return ans

arr=list(map(int,input().split()))
ans=0
if arr[0]!=0:
    arr[0]-=1
    for i in range(2):
        num=str(arr[i])
        n=len(num)
        prev=10
        tight=1
        trail=1
        dp=[[[[-1 for  i in range(2)]for l in range(2)]for j in range(11)]for k in range(n+1)]
        ans=counting_numbers(n,prev,tight,trail,num,dp)-ans
else:
    num = str(arr[1])
    n = len(num)
    prev = 10
    tight = 1
    trail = 1
    dp = [[[[-1 for i in range(2)] for l in range(2)] for j in range(11)] for k in range(n + 1)]
    ans = counting_numbers(n, prev, tight, trail, num, dp) - ans
print(ans)





























