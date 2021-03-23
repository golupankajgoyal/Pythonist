from collections import defaultdict
from math import inf

# Job Assignment Problem
# Let there be N workers and N jobs. Any worker can be assigned to perform any job,
# incurring some cost that may vary depending on the work-job assignment.
# It is required to perform all jobs by assigning exactly one worker to each job
# and exactly one job to each agent in such a way that the total cost of the assignment is minimized.
def assign_jobs(n,i,mask,dp,cost):
    if i==n:
        return 0
    if dp[i][mask]!=inf:
        return dp[i][mask]
    for j in range(n):
        if mask&(1<<j):
            dp[i][mask]=min(dp[i][mask],cost[i][j]+assign_jobs(n,i+1,mask&(~(1<<j)),dp,cost))
    return dp[i][mask]

# cost = [[9, 2, 7, 8],
#         [6, 4, 3, 7],
#         [5, 8, 1, 8],
#         [7, 6, 9, 4]]
# n=4
# mask=(1<<n)-1
# dp=[[inf for i in range(1<<n)]for j in range(n)]
# print(assign_jobs(n,0,mask,dp,cost))
# for i in range(n):
#     print(dp[i])

# Traveling Salesman Problem (TSP)
# https://www.geeksforgeeks.org/travelling-salesman-problem-set-1/
# Using bitmask+DP O((2^n)n)
def travelling_salesman(city,mask,distances,dp):
    if city==len(distances):
        return 0
    if dp[city][mask]!=inf:
        return dp[city][mask]
    temp=inf
    for i in range(len(distances)):
        if mask&(1<<i) and city!=i:
            temp=min(temp,travelling_salesman(city+1,mask&(~(1<<i)),distances,dp)+distances[city][i])
    dp[city][mask]=temp
    return temp
# distance = [[0, 10, 15, 20], [10, 0, 35, 25],
#             [15, 35, 0, 30], [20, 25, 30, 0]]
# mask=(1<<len(distance))-1
# dp=[[inf for i in range(1<<len(distance))]for j in range(len(distance))]
# print(travelling_salesman(0,mask,distance,dp))
# for i in range(len(distance)):
#     print(dp[i])

# Little Elephant and T-Shirts
def ways_to_wear_tshirts(i,mask,dp,n,tshirts):
    if mask==0:
        return 1
    if i>100 :
        return 0
    if dp[i][mask]!=-1:
        return dp[i][mask]
    # print(bin(mask),i)
    ans=0
    for person in range(n):
        if tshirts[person][i]==1 and mask&(1<<person):
            ans+=ways_to_wear_tshirts(i+1,mask&(~(1<<person)),dp,n,tshirts)
    ans+=ways_to_wear_tshirts(i+1,mask,dp,n,tshirts)
    dp[i][mask]=ans
    return ans

# people=defaultdict((list))
# people[0]=[5, 100, 1]
# people[1]=[2]
# people[2]=[5, 100]
# n=len(people)
# dp=[[-1 for j in range(1<<n)] for i in range(100+1)]
# mask=(1<<n)-1
# print(ways_to_wear_tshirts(1,mask,dp,n,people))
# for i in range(100+1):
#     print(dp[i])

#The Ghost Type
# https://www.hackerearth.com/practice/algorithms/dynamic-programming/bit-masking/practice-problems/algorithm/the-ghost-type/

def isValid(j,mask,n):
    for k in range(n):
        if (k+1)!=j and mask&(1<<k) and j&(k+1)==(k+1):
            return False
    return True

def ghost_type(i,mask,dp,n):
    if mask==0:
        return 1
    if i>n:
        return 0
    if dp[i][mask]!=-1:
        return dp[i][mask]
    ans=0
    for j in range(n):
        if mask&(1<<j) and isValid(j+1,mask,n):
            ans+=ghost_type(i+1,mask&(~(1<<(j))),dp,n)
    dp[i][mask]=ans
    return ans
# for n=15 ans=1680384
# n=4
# dp=[[-1 for i in range(1<<n)]for j in range(n+1)]
# mask=(1<<n)-1
# print(ghost_type(1,mask,dp,n))

# Counting Tiles
# Your task is to count the number of ways you can fill an n×m grid using 1×2 and 2×1 tiles.
def generate_new_masks(row,next_mask,mask,n,new_masks):
    if row==n:
        new_masks.append(next_mask)
        return
    if mask&(1<<row):
        generate_new_masks(row+1,next_mask&(~(1<<row)),mask,n,new_masks)
    if mask&(1<<row)==0:
        generate_new_masks(row+1,next_mask,mask,n,new_masks)
    # print(row, bin(mask), bin(next_mask),n-1,mask&(1<<row))
    if row<n-1 and mask&(1<<row) and mask&(1<<(row+1)):
        generate_new_masks(row+2,next_mask,mask,n,new_masks)

def count_tiling(col,mask,n,m,dp):
    if col==m:
        if mask==(1<<n)-1:
            return 1
        else:
            return 0
    if dp[col][mask]!=-1:
        return dp[col][mask]
    new_masks=[]
    generate_new_masks(0,(1<<n)-1,mask,n,new_masks)
    ans=0
    for new_mask in new_masks:
        ans+=count_tiling(col+1,new_mask,n,m,dp)
    dp[col][mask]=ans
    return ans
n,m=4,7
mask=(1<<n)-1
dp=[[-1 for i in range(mask+1)]for j in range(m+1)]
print(count_tiling(0,mask,n,m,dp))





























