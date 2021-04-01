from math import inf,log
from collections import defaultdict,deque
from heapq import heapify,heappush as hpush,heappop as hpop,heapreplace as hreplace
import sys
# Dice Combinations
# Your task is to count the number of ways to construct sum n by throwing a dice one or more times.
# Each throw produces an outcome between 1 and 6.
md=(10**9)+7
sys.setrecursionlimit(1 << 30)
def dice_combinations(num,dp):
    if num<0:
        return 0
    if num==0:
        return 1
    if dp[num]!=-1:
        return dp[num]
    ans=0
    for  i in range(1,7):
        ans+=dice_combinations(num-i,dp)
    dp[num]=ans
    return ans

# num=int(input())
# dp=[-1 for i in range(num+1)]
# print(dice_combinations(num,dp)%mod)

# Minimizing Coins
# Consider a money system consisting of n coins. Each coin has a positive integer value.
# Your task is to produce a sum of money x using the available coins in such a way that the
# number of coins is minimal.
def min_coin(x,coins,dp):
    if x==0:
        return 0
    if dp[x]!=-1:
        return dp[x]
    ans=inf
    for coin in coins:
        if x-coin>=0:
            ans=min(ans,min_coin(x-coin,coins,dp))
    dp[x]=ans+1
    return dp[x]

# n,x=map(int,input().split())
# coins=list(map(int,input().split()))
# dp=[-1 for i in range(x+1)]
# min_coin(x,coins,dp)
# print(dp[x] if dp[x]!=inf else -1)

# Coin Combinations I
# Consider a money system consisting of n coins. Each coin has a positive integer value.
# Your task is to calculate the number of distinct ways you can produce a money sum x using the available coins.
def min_coin(x,coins,dp):
    if x==0:
        return 1
    if dp[x]!=-1:
        return dp[x]
    ans=0
    for coin in coins:
        if x-coin>=0:
            ans+=min_coin(x-coin,coins,dp)
    dp[x]=ans
    return dp[x]
# n,x=map(int,input().split())
# coins=list(map(int,input().split()))
# dp=[-1 for i in range(x+1)]
# min_coin(x,coins,dp)
# print(dp[x])

# Coin Combinations II
# Consider a money system consisting of n coins. Each coin has a positive integer value.
# Your task is to calculate the number of distinct ordered ways you can produce a money sum x
# using the available coins.
# https://cses.fi/problemset/task/1636
def min_coin(x,coins,dp,prev):
    # print(x,prev)
    if x<0:
        return 0
    if x==0:
        return 1
    if dp[x][prev]!=-1:
        return dp[x][prev]
    ans=0
    for curr in range(prev,len(coins)):
        temp=coins[curr]
        ans+=min_coin(x-temp,coins,dp,curr)
    dp[x][prev]=ans
    return ans
# n,x=map(int,input().split())
# coins=list(map(int,input().split()))
# coins=sorted(coins)
# dp=[[-1 for j in range(n+1)]for i in range(x+1)]
# min_coin(x,coins,dp,0)
# print(dp[x][0])

#Removing Digits
# You are given an integer n. On each step, you may substract from it any one-digit number that appears in it.
# How many steps are required to make the number equal to 0?
def remove_digits(n,dp):
    if n<10:
        return 1
    # print(n)
    if dp[n]!=-1:
        return dp[n]
    temp=n
    ans=inf
    while temp>0:
        num=temp%10
        temp//=10
        if num>0:
            ans=min(ans,remove_digits(n-num,dp))
    dp[n]=ans+1
    return dp[n]

# n=int(input())
# dp=[-1 for  i in range(n+1)]
# print(remove_digits(n,dp))

# Grid Paths
# Consider an n×n grid whose squares may have traps. It is not allowed to move to a square with a trap.
# Your task is to calculate the number of paths from the upper-left square to the lower-right square
# where you only can move right or down.
def find_paths(row,col,grid,dp,n):
    # print(row,col)
    if row==n-1 and col==n-1 :
        if grid[row][col]!="*":
            return 1
        else:
            return 0
    if row>=n or col>=n or grid[row][col]=="*":
        return 0
    if dp[row][col]!=-1:
        return dp[row][col]
    paths=find_paths(row,col+1,grid,dp,n)
    paths+=find_paths(row+1,col,grid,dp,n)
    dp[row][col]=paths%md
    return dp[row][col]

# n=int(input())
# grid=[]
# for i in range(n):
#     ins = [x for x in input()]
#     grid.append(ins)
# dp=[[-1 for i in range(n+1)]for j in range(n+1)]
# print(find_paths(0,0,grid,dp,n))

#Book Shop
# You are in a book shop which sells n different books. You know the price and number of pages of each book.
# You have decided that the total price of your purchases will be at most x.
# What is the maximum number of pages you can buy? You can buy each book at most once.
def max_pages(pos,money,price,pages,dp):
    if pos>=len(price) or money<=0:
        return 0
    if dp[pos][money]!=-1:
        return dp[pos][money]
    count=max_pages(pos+1,money,price,pages,dp)
    if money-price[pos]>=0:
        count=max(count,max_pages(pos+1,money-price[pos],price,pages,dp)+pages[pos])
    dp[pos][money]=count
    return count

# n,money=map(int,input().split())
# price=list(map(int,input().split()))
# pages=list(map(int,input().split()))
# dp=[[-1 for i in range(money+1)]for j in range(n+1)]
# print(max_pages(0,money,price,pages,dp))

#Array Description
# You know that an array has n integers between 1 and m, and the difference between two adjacent values
# is at most 1.
# Given a description of the array where some values may be unknown,
# your task is to count the number of arrays that match the description.

def find_valid_arr(pos,num,m,arr):
    if num>m or num<1:
        return 0
    if arr[pos]!=0 and arr[pos]!=num:
        return 0
    if dp[pos][num]!=-1:
        return dp[pos][num]
    if pos==len(arr)-1:
        if arr[pos]==0 or arr[pos]==num:
            return 1
        else:
            return 0
    result=find_valid_arr(pos+1,num-1,m,arr)+find_valid_arr(pos+1,num,m,arr)+find_valid_arr(pos+1,num+1,m,arr)
    dp[pos][num]=result
    return result

# n,m=map(int,input().split())
# arr=list(map(int,input().split()))
# dp=[[-1 for i in range(m+1)]for j in range(n+1)]
# result=0
# for i in range(1,m+1):
#     result+=find_valid_arr(0,i,m,arr)
# print(result)

# Counting Towers
# Your task is to build a tower whose width is 2 and height is n.
# You have an unlimited supply of blocks whose width and height are integers.
# https://cses.fi/problemset/task/2413
# Resource : https://www.youtube.com/watch?v=pMEYMYTX-r0&list=PLb3g_Z8nEv1h1w6MI8vNMuL_wrI0FtqE7&index=28
def counting_towers(pos,link,dp,n):
    if pos>n:
        return 1
    result=0
    if dp[link][pos]!=-1:
        return dp[link][pos]
    if link==0:
        temp=counting_towers(pos+1,link,dp,n)
        result+=temp
        result+=(temp*2)
        result+=(temp+counting_towers(pos+1,1,dp,n))
    else:
        temp=counting_towers(pos+1,link,dp,n)
        result+=temp
        result+=(counting_towers(pos+1,0,dp,n)+temp)
    dp[link][pos]=result
    return result

# t=int(input())
# for _ in range(t):
#     n=int(input())
#     dp=[[-1 for i in range(n+1)]for j in range(2)]
#     print(counting_towers(2,0,dp,n)+counting_towers(2,1,dp,n))

# Edit Distance
# https://cses.fi/problemset/task/1639
def edit_distance(p1,p2,s1,s2,dp):
    if p1>=len(s1):
        return len(s2)-p2
    if p2>=len(s2):
        return len(s1)-p1
    if dp[p1][p2]!=-1:
        return dp[p1][p2]
    if s1[p1]==s2[p2]:
        result=edit_distance(p1+1,p2+1,s1,s2,dp)
    else:
        result=min(edit_distance(p1+1,p2+1,s1,s2,dp),
                   edit_distance(p1+1,p2,s1,s2,dp),
                   edit_distance(p1,p2+1,s1,s2,dp))+1
    dp[p1][p2]=result
    return result
# s1=input()
# s2=input()
# dp=[[-1 for i in range(len(s2)+1)]for j in range(len(s1)+1)]
# print(edit_distance(0,0,s1,s2,dp))

# Rectangle Cutting
# Given an a×b rectangle, your task is to cut it into squares. On each move you can select a rectangle
# and cut it into two rectangles in such a way that all side lengths remain integers.
# What is the minimum possible number of moves?
def rect_cutting(a,b,dp):
    if a==b:
        return 0
    if dp[a][b]!=-1:
        return dp[a][b]
    ans=inf
    for i in range(1,b//2+1):
        ans=min(ans,rect_cutting(a,b-i,dp)+rect_cutting(a,i,dp))
    for i in range(1,a//2+1):
        ans = min(ans, rect_cutting(a-i, b, dp) + rect_cutting(i, b, dp))
    ans+=1
    dp[a][b]=ans
    # print(a,b,ans)
    return ans
# a,b=map(int,input().split())
# if a>b:
#     a,b=b,a
# dp=[[-1 for i in range(b+1)]for j in range(a+1)]
# print(rect_cutting(a,b,dp))

# Money Sums
# You have n coins with certain values. Your task is to find all money sums you can create using these coins.
def money_sum(pos,coins,ans):
    if pos>=len(coins):
        return
    money_sum(pos+1,coins,ans)
    prev=list(ans)
    for num in prev:
        ans.add(num+coins[pos])
    return

# n=int(input())
# coins=list(map(int,input().split()))
# ans=set()
# ans.add(0)
# money_sum(0,coins,ans)
# print(len(ans)-1)
# ans=sorted(ans)
# for num  in ans:
#     if num!=0:
#         result.append(num)
#         print(num,end=" ")

# Removal Game
# There is a list of n numbers and two players who move alternately. On each move,
# a player removes either the first or last number from the list, and their score increases by that number.
# Both players try to maximize their scores.
# What is the maximum possible score for the first player when both players play optimally?
def  removal_game(start,end,arr,dp):
    if end==start:
        return arr[start]
    if end-start==1:
        return max(arr[start],arr[end])
    if dp[start][end]!=-1:
        return dp[start][end]
    result1=arr[start]+min(removal_game(start+2,end,arr,dp),removal_game(start+1,end-1,arr,dp))
    result2=arr[end]+min(removal_game(start,end-2,arr,dp),removal_game(start+1,end-1,arr,dp))
    dp[start][end]=max(result2,result1)
    return dp[start][end]

# n=int(input())
# arr=list(map(int,input().split()))
# dp=[[-1 for i in range(n+1)]for j in range(n+1)]
# print(removal_game(0,n-1,arr,dp))

# Two Sets II
# Your task is to count the number of ways numbers 1,2,…,n can be divided into two sets of equal sum.
def divide_sets(pos,sum,n,num,dp):
    if sum==num:
        return 1
    if pos>n:
        return 0
    if sum>num:
        return 0
    if dp[pos][sum]!=-1:
        return dp[pos][sum]
    count=divide_sets(pos+1,sum+pos,n,num,dp)
    count=(count+divide_sets(pos+1,sum,n,num,dp))%md
    dp[pos][sum]=count
    return count

# n=int(input())
# sum=(n*(n+1))//2
# if sum%2!=0:
#     print(0)
# else:
#     # print(sum)
#     dp=[[-1 for i in range(sum+1)]for j in range(n+1)]
#     print((divide_sets(1,0,n,sum//2,dp)*500000004)%md)

# Increasing Subsequence
# You are given an array containing n integers. Your task is to determine the longest increasing
# subsequence in the array, i.e., the longest subsequence where every element is larger than the previous one.
def lcs(i,arr,dp):
    if dp[i]!=-1:
        return dp[i]
    temp = 0
    for j in range(i-1,-1,-1):
        if arr[i]>arr[j]:
            temp=max(temp,lcs(j,arr,dp))
    temp+=1
    dp[i]=temp
    return temp
# n=int(input())
# arr=list(map(int,input().split()))
# dp=[-1  for i in range(n)]
# for i in range(n-1,-1,-1):
#     if dp[i]==-1:
#         lcs(i,arr,dp)
# print(max(dp))

# Projects
# There are n projects you can attend. For each project, you know its starting and ending days
# and the amount of money you would get as reward. You can only attend one project during a day.
# What is the maximum amount of money you can earn?
def find_next(num,projects):
    start=0
    end=len(projects)-1
    while start!=end:
        # print(start,end)
        mid = (start + end) // 2
        if projects[mid][0]>num:
            end=mid
        else:
            start=mid+1
    # print(start,end)
    return start if projects[start][0]> num else len(projects)

def max_earning(pos,projects,dp):
    if pos>=len(projects):
        return 0
    if dp[pos]!=-1:
        return dp[pos]
    earn=max_earning(pos+1,projects,dp)
    next_pos=find_next(projects[pos][1],projects)
    earn=max(earn,max_earning(next_pos,projects,dp)+projects[pos][2])
    dp[pos]=earn
    return earn

# n=int(input())
# projects=[]
# for _ in range(n):
#     p=[0]*3
#     p[0],p[1],p[2]=map(int,input().split())
#     projects.append(p)
# projects=sorted(projects,key=lambda x:x[0])
# dp=[-1 for  i in range(n+1)]
# # print(projects)
# print(max_earning(0,projects,dp))
# print(dp)

# Elevator Rides
# There are n people who want to get to the top of a building which has only one elevator.
# You know the weight of each person and the maximum allowed weight in the elevator.
# What is the minimum number of elevator rides?

# My Solution
# def add_masks(mask,pos,capacity,masks,weights,visited):
#     print(mask,pos,capacity)
#     if capacity<0:
#         return
#     if mask==0:
#         masks.add(0)
#         return
#     if pos==len(weights) :
#         if visited[mask]==False:
#             masks.add(mask)
#         return
#     add_masks(mask,pos+1,capacity,masks,weights,visited)
#     if mask&(1<<pos)!=0:
#         add_masks(mask&(~(1<<pos)),pos+1,capacity-weights[pos],masks,weights,visited)
#     return
#
# def elevator_rides(capacity,weights,n):
#     mask=(1<<n)-1
#     old_masks={mask}
#     turn=0
#     visited=[False for i in range(mask+1)]
#     while True:
#         for  mask in old_masks:
#             print(bin(mask),end=" ")
#         new_masks=set()
#         turn+=1
#         for mask in old_masks:
#             visited[mask]=True
#             add_masks(mask,0,capacity,new_masks,weights,visited)
#             if  0 in new_masks:
#                 return turn
#         old_masks=new_masks

# Books Solution(Competetive Coding)
def elevator_rides(mask,weights,dp,capacity,n):
    if mask==0:
        return [1,0]
    if dp[mask]!=-1:
        return dp[mask]
    ans=[n+1,0]
    for i in range(n):
        if mask&(1<<i)!=0:
            temp=list(elevator_rides(mask&(~(1<<i)),weights,dp,capacity,n))

            if temp[1]+weights[n-i-1]<=capacity:
                temp[1]+=weights[n-i-1]

            else:
                temp[0]+=1
                temp[1]=weights[n-i-1]
            ans=min(ans,temp)
    dp[mask]=ans
    return ans

# n,capacity=map(int,input().split())
# weights=list(map(int,input().split()))
# mask=(1<<n)-1
# dp=[-1 for i in range(1<<n)]
# # print(dp)
# print(elevator_rides(mask,weights,dp,capacity,n))

#Beautiful Array
# https://codeforces.com/contest/1155/problem/D
def max_sum_subarr(arr,index):
    max_sum=0
    start=end=0
    curr_sum=0
    while end<len(arr):
        if curr_sum+arr[end]>0:
            curr_sum+=arr[end]
            if curr_sum>max_sum:
                max_sum=curr_sum
                index[0],index[1]=start,end
        else:
            curr_sum=0
            start=end+1
        end+=1
    return max_sum

def convert_arr(arr,n):
    for i in range(n):
        arr[i]*=-1

def beautiful_arr(arr,x,n):
    pos_index=[-1,-1]
    pos_sum=max_sum_subarr(arr,pos_index)
    convert_arr(arr,n)
    neg_index=[-1,-1]
    neg_sum=max_sum_subarr(arr,neg_index)
    convert_arr(arr,n)
    if x<0 and neg_index[0]!=-1:
        for i in range(neg_index[0],neg_index[1]+1):
            arr[i]=arr[i]*x
        neg_sum=max_sum_subarr(arr,neg_index)
        return max(neg_sum, pos_sum)
    elif x>0:
        pos_sum*=x
    return pos_sum

n,x=map(int,input().split())
arr=list(map(int,input().split()))
print(beautiful_arr(arr,x,n))































