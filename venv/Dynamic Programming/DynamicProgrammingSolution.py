import sys
from sys import maxsize
from math import inf
from collections import deque
from operator import itemgetter,attrgetter
def memFindSumOfSquares(n,arr):
    if n<=1:
        arr[n]=n
        return n
    ans=100000
    for i in range(1,int(n**(1/2))+1):
        check=n-i**2
        if arr[check]==-1:
            ans=min(ans,memFindSumOfSquares(check,arr))
        else:
            ans=min(arr[check],ans)
    arr[n]=ans+1
    return arr[n]

def dpFindSquareForSum(n):
    arr=[-1]*(n+1)
    arr[0]=0
    arr[1]=1
    for i in range(2,n+1):
        count=100000
        for j in range(1,int(i**(0.5)+1)):
            temp=i-j*j
            count=min(count,arr[temp])
        arr[i]=count+1
    print(arr[n])

def memCountOfBalanceBT(h,arr):
    if h<=1:
        return 1
    MOD = 1000000007
    if arr[h-1]==-1:
        x=memCountOfBalanceBT(h-1,arr)
    else:
        x=arr[h-1]
    if arr[h-2]==-1:
        y=memCountOfBalanceBT(h-2,arr)
    else:
        y=arr[h-2]
    ans=((x*x)%MOD+2*((x*y)%MOD))%MOD
    arr[h]=ans
    return ans

def dpCountOFBalanceBT(h):
    MOD=1000000007
    arr=[-1]*(h+1)
    arr[0]=1
    arr[1]=1
    for i in range(2,h+1):
        x=arr[i-1]
        y=arr[i-2]
        ans=((x*x)%MOD+2*((x*y)%MOD))%MOD
        arr[i]=ans
    print(arr[h])

def minCostPath(i,j,m,n,cost,dp):
    if i>=m or j>=n:
        return sys.maxsize
    if dp[i][j]!=sys.maxsize:
        return dp[i][j]
    if i==m-1 and j==n-1:
        return cost[m-1][n-1]
    x=minCostPath(i,j+1,m,n,cost,dp)
    if x!=sys.maxsize:
        dp[i][j+1]=x
    y=minCostPath(i+1,j,m,n,cost,dp)
    if y!=sys.maxsize:
        dp[i+1][j]=y
    z=minCostPath(i+1,j+1,m,n,cost,dp)
    if z!=sys.maxsize:
        dp[i+1][j+1]=z
    dp[i][j]=min(x,y,z)+cost[i][j]
    return dp[i][j]

def dpMinCostPath(cost,m,n):
    dp=[[-1 for j in range(n)]for i in range(m)]
    dp[m-1][n-1]=cost[m-1][n-1]

    for j in range(m-2,-1,-1):
        dp[j][n-1]=dp[j+1][n-1] + cost[j][n-1]

    for i in range(n-2,-1,-1):
        dp[m-1][i]=dp[m-1][i+1] + cost[m-1][i]
    for i in range(m-2,-1,-1):
        for j in range(n-2,-1,-1):
            dp[i][j]=min(dp[i+1][j],dp[i+1][j+1],dp[i][j+1]) + cost[i][j]
    print(dp)

def findLCS(s,t):
    if len(s)==0 or len(t)==0:
        return 0
    if s[0]==t[0]:
        return findLCS(s[1:],t[1:])+1
    a=findLCS(s[1:],t)
    b=findLCS(s,t[1:])
    return max(a,b)
def memFindLCS(s,t,dp):
    if len(s)==0 or len(t)==0:
        print(len(s),len(t))
        return 0
    sLen=len(s)
    tLen=len(t)
    if dp[sLen][tLen]!=-1:
        return dp[sLen][tLen]
    if s[0]==t[0]:
        dp[sLen-1][tLen-1]=memFindLCS(s[1:],t[1:],dp)
        dp[sLen][tLen]=dp[sLen-1][tLen-1]+1
        return dp[sLen][tLen]
    dp[sLen-1][tLen]=memFindLCS(s[1:],t,dp)
    dp[sLen][tLen-1]=memFindLCS(s,t[1:],dp)
    dp[sLen][tLen]=max(dp[sLen-1][tLen],dp[sLen][tLen-1])
    return dp[sLen][tLen]

def dpFindLCS(s,t):
    l1=len(s)
    l2=len(t)
    dp=[[0 for i in range(l2+1)]for j in range(l1+1)]
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            if s[l1-i]==t[l2-j]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    print(dp)


def editDistance(s,t):
    if len(s)==0 or len(t)==0:
        return max(len(s),len(t))
    if s[0]==t[0]:
        return editDistance(s[1:],t[1:])

    a=editDistance(s[1:],t)
    b=editDistance(s,t[1:])
    c=editDistance(s[1:],t[1:])
    return min(a,b,c)+1



def memEditDistance(s,t,dp):
    if len(s)==0 or len(t)==0:
        return max(len(s),len(t))
    l1=len(s)
    l2=len(t)
    if dp[l1][l2]!=-1:
        return dp[l1][l2]
    if s[0]==t[0]:
        dp[l1][l2]=memEditDistance(s[1:],t[1:],dp)
    else:
        dp[l1-1][l2]=memEditDistance(s[1:],t,dp)
        dp[l1][l2-1]=memEditDistance(s,t[1:],dp)
        dp[l1-1][l2-1]=memEditDistance(s[1:],t[1:],dp)
        dp[l1][l2] = min(dp[l1 - 1][l2],dp[l1][l2 - 1],dp[l1 - 1][l2 - 1])+1
    return dp[l1][l2]

def dpEditDistance(s,t):
    l1=len(s)
    l2=len(t)
    dp=[[-1 for j in range(l2 +1)]for i in range(l1 +1)]
    for i in range(l1+1):
        dp[i][0]=i
    for i in range(l2+1):
        dp[0][i]=i
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            if s[l1-i]==t[l2-j]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1

    for i in range(len(s)+1):
        print(dp[i])


def alphaCode(code,size,ans):
    if ans[size]>0:
        return ans[size]
    if size==1:
        ans[size]=1
        return 1
    if size==0:
        ans[size]=1
        return 1

    output=alphaCode(code,size-1,ans)
    if (int(code[size-2])*10+int(code[size-1]))<=26:
        output+=alphaCode(code,size-2,ans)
    ans[size]=output
    return output

def alphaCodeInit(code):
    ans=[0]*(len(code)+1)
    output=alphaCode(code,len(code),ans)
    print(ans)
    return output

def LIS(input,size,ans):
    if size==1:
        ans[1]=1
        return 1
    if ans[size-1]==0:
        LIS(input,size-1,ans)
    maxLen=0
    for i in range(size):
        if input[i]<input[size-1]:
            maxLen=max(maxLen,ans[i+1])
    ans[size]=maxLen+1
    return ans[size]

def LISInit(input):
    ans=[0]*(len(input)+1)
    output=LIS(input,len(input),ans)
    print(ans)
    return output

def stairCase(output,n):
    if n==0:
        return 1
    if n==1:
        return 1
    if n==2:
        return 2
    output[n]+=(stairCase(output,n-1))
    output[n]+=(output[n-2])
    output[n]+=(output[n-3])
    return output[n]

def stairCaseInit(n):
    output=[0]*(n+1)
    output[0],output[1],output[2]=1,1,2
    print(stairCase(output,6))

def coinChange(amount,coins,dp):
    if amount==0:
        dp[0][len(coins)]=1
        return 1
    if amount<0:
        return 0
    if len(coins)==0:
        dp[amount][len(coins)]=0
        return 0
    if dp[amount][len(coins)]<0:
        withCoin=coinChange(amount-coins[0],coins,dp)
        withoutCoin=coinChange(amount,coins[1:],dp)
        dp[amount][len(coins)]=withCoin+withoutCoin
    return dp[amount][len(coins)]

def coinChngeInit(amount,coins):
    dp=[[-1 for j in range(len(coins)+1)]for i in range(amount+1)]
    print(coinChange(amount,coins,dp))
    for i in range(amount+1):
        print(dp[i])

def kadaneAlgo(arr):
    maxSoFar=0
    currMax=0
    for num in arr:
        currMax+=num
        if currMax>maxSoFar:
            maxSoFar=currMax
        if currMax<0:
            currMax=0
    print(maxSoFar)

def magicGrid(grid,sr,sc,er,ec,dp):
    if sr==er and sc==ec:
        dp[er][ec]=1
        return 1
    if dp[sr][sc]!=-1:
        return dp[sr][sc]
    right=maxsize
    bottom=maxsize
    if sc+1<=ec:
        right=magicGrid(grid,sr,sc+1,er,ec,dp)-grid[sr][sc+1]
    if sr+1<=er:
        bottom=magicGrid(grid,sr+1,sc,er,ec,dp)-grid[sr+1][sc]
    dp[sr][sc]=max(min(right,bottom),1)
    return dp[sr][sc]

def magicGridInit(grid):
    dp=[[-1 for j in range(1)]for i in range(len(grid))]
    print(max(magicGrid(grid,0,0,6,0,dp)-grid[0][0],1))

def knapsack(w,v,c):
    if c<=0 or len(w)==0:
        return 0
    a=0
    if c-w[0]>=0:
        a=v[0] + knapsack(w[1:],v[1:],c-w[0])
    b=knapsack(w[1:],v[1:],c)
    return max(a,b)

def knapsackMem(wt,v,minWt,dp,count):
    count[0]+=1
    if len(wt)==0 or minWt==0:
        return 0
    if dp[minWt][len(wt)]!=-1:
        return dp[minWt][len(wt)]
    withWt=0
    if minWt-wt[0]>=0:
        withWt=knapsackMem(wt[1:],v[1:],minWt-wt[0],dp,count)+v[0]
    withoutWt=knapsackMem(wt[1:],v[1:],minWt,dp,count)
    dp[minWt][len(wt)]=max(withWt,withoutWt)
    return dp[minWt][len(wt)]

def knapsackDp(wt,val,W,n):
    dp=[[-1 for j in range(n+1)]for i in range(W+1)]
    for i in range(n+1):
        dp[0][i]=0
    for j in range(W+1):
        dp[j][0]=0
    for i in range(1,W+1):
        for j in range(1,n+1):
            withWt=0
            if i-wt[n-j]>=0:
                withWt=val[n-j]+dp[i-wt[n-j]][j-1]
            dp[i][j]=max(dp[i][j-1],withWt)
    for i in range(W+1):
        print(dp[i])

def findSumPossible(n,values,dp):
    print(n,len(values))
    if n==0 :
        dp[len(values)][0]=1
        return 1
    if len(values)==0:
        dp[0][n]=0
        return 0
    if dp[len(values)][n]!=-1:
        return dp[len(values)][n]
    dp[len(values)][n]=findSumPossible(n,values[1:],dp)
    if n-values[0]>=0:
        dp[len(values)][n]=findSumPossible(n-values[0],values[1:],dp) or dp[len(values)][n]
    return dp[len(values)][n]

def findSumPossibleDp(num,values):
    dp=[[-1 for j in range(num+1)]for i in range(len(values)+1)]
    for i in range(num+1):
        dp[0][i]=0
    for i in range(len(values)+1):
        dp[i][0]=1
    for i in range(1,len(values)+1):
         for j in range(1,num+1):
             dp[i][j]= dp[i-1][j]
             if j>=values[len(values)-i]:
                dp[i][j]=dp[i][j] or dp[i-1][j-values[len(values)-i]]
    for i in range(len(values)+1):
        print(dp[i])

def findSumPossibleDpWithXor(num,values):
    dp=[[-1 for j in range(num+1)]for i in range(2)]
    for i in range(num+1):
        dp[0][i]=0
    for i in range(2):
        dp[i][0]=1
    flag=1
    for i in range(0,len(values)):
        for j in range(1,num+1):
            dp[flag][j]= dp[flag^1][j]
            if j>=values[i]:
                dp[flag][j]=dp[flag][j] or dp[flag^1][j-values[i]]
        flag^=1
    for i in range(2):
        print(dp[i])
    print(dp[flag^1][num])

# Two water Jug problem
# You are at the side of a river. You are given a m litre jug and a n litre jug where 0 < m < n.
# Both the jugs are initially empty. The jugs don’t have markings to allow measuring smaller quantities.
# You have to use the jugs to measure d litres of water where d < n.
# Determine the minimum no of operations to be performed to obtain d litres of water in one of jug.
# The operations you can perform are:
# 1. Empty a Jug
# 2. Fill a Jug
# 3. Pour water from one jug to the other until one of the jugs is either empty or full.
# User function Template for python3
# Without dp
def gcd(n, m):
    if m == 0:
        return n
    return gcd(m, n % m)

def count_steps(from_jug, to_jug, d):
    from_jug_level = from_jug
    to_jug_level = 0
    step = 1
    while from_jug_level != d and to_jug_level != d:
        temp = min(from_jug_level, to_jug - to_jug_level)
        from_jug_level -= temp
        to_jug_level += temp
        step += 1
        # print(from_jug_level,to_jug_level,step)
        if from_jug_level == d or to_jug_level == d:
            # print(step)
            return step

        if to_jug_level == to_jug:
            to_jug_level = 0

            step += 1
            # print(from_jug_level,to_jug_level,step)

        if from_jug_level == 0:
            from_jug_level = from_jug

            step += 1
            # print(from_jug_level,to_jug_level,step)
    # print(step)
    return step

def minSteps(self, m, n, d):
    # Code here
    if gcd(n, m) // d != 0:
        return -1
    count = min(count_steps(m, n, d), count_steps(n, m, d))
    return count

# Using Recursion
def find_min_steps(i,j,m,n,d):
    if i==d or j==d:
        return 0
    temp=min(i,n-j)
    step=0
    i-=temp
    j+=temp
    step+=1
    print(i, j)
    if i==d or j==d:
        return 2
    if i==0:
        i=m
        step += 1
        print(i, j)
    if j==n:
        j=0
        step += 1
        print(i, j)
    return step + find_min_steps(i,j,m,n,d)
# m = 3
# n = 5
# d = 4
# print(3,0)
# print(find_min_steps(3,0,m,n,d))

# Determine the minimum time needed to search through a set of folders given a number of workers.
# The workers must search a contiguous set of folders, and each folder may take a
# varying amount of time to process.
def  find_min_load(workers,folders,dp,count):
    count[0] += 1
    if len(folders)==0:
        return 0
    if workers==1:
        dp[1][len(folders)]=sum(folders)
        return dp[1][len(folders)]
    if len(folders)<workers:
        dp[workers][len(folders)]=-1
        return -1
    if dp[workers][len(folders)]!=-1:
        return dp[workers][len(folders)]
    min_load=inf
    for i in range(len(folders)):
        curr_load=sum(folders[:i+1])
        if dp[workers-1][len(folders[i+1:])]==-1:
            find_min_load(workers-1,folders[i+1:],dp,count)
        temp=dp[workers-1][len(folders[i+1:])]
        min_load=min(min_load,max(curr_load,temp))
    dp[workers][len(folders)]=min_load
    return min_load

# arr=[ 568, 712, 412, 231, 241, 393, 865, 287, 128, 457, 238, 98, 980, 23, 782]
# workers=4
# count=[0]
# dp=[[-1 for i in range(len(arr)+1)]for j in range(workers+1)]
# print(find_min_load(workers,arr,dp,count))
# for i in range(len(dp)):
#     print(dp[i])
# print(count[0])
# Returns: 1785
# The filing cabinets should be partitioned as follows:
# 	568 712 412 | 231 241 393 865 | 287 128 457 238 98 | 980 23 782


# https://www.hackerrank.com/challenges/equal/problem
def count_step(arr, num):
    result = -1
    for i in arr:
        x=i-num
        temp = x // 5 + (x % 5) // 2 + (x % 5) % 2
        result +=temp
    return result


def equal(arr):
    min_count = min(arr)
    ans = inf
    print(min_count)
    for i in range(max(0, min_count - 4), min_count + 1):
        temp=count_step(arr, i)
        ans = min(ans,temp )
        print(i,temp)
    return ans+1

# arr=[520,862,10,956,498,956,991,542,523,664,378,194,76,90,753,868,837,830,932,814,616,78,103,882,452,397,899,488,149,108,723,22,323,733,330,821,41,322,715,917,986,93,111,63,535,864,931,372,47,215,539,15,294,642,897,98,391,796,939,540,257,662,562,580,747,893,401,789,215,468,58,553,561,169,616,448,385,900,173,432,115,712]
# print(equal(arr))

def longest_bitonic_subsequence(arr,i,dp):
    if i ==0:
        dp[0]=[1,1]
        return

    if dp[i-1]==-1:
        longest_bitonic_subsequence(arr, i-1, dp)
    max_seq=0
    order=0
    for j in range(i-1,-1,-1):
        if arr[i]>arr[j]:
            if dp[j][1]==1 and max_seq<dp[j][0]:
                max_seq=dp[j][0]
                order=1
        elif  arr[i]<arr[j]:
            if max_seq<dp[j][0]:
                max_seq =dp[j][0]
                order=-1
        else:
            max_seq = dp[j][0]
            order = dp[j][1]
            break
    if order==0:
        dp[i]=[1,1]
    else:
        dp[i]=[max_seq+1,order]

def longest_bitonic_subsequence_init(arr):
    dp = [-1 for i in range(len(arr))]
    longest_bitonic_subsequence(arr, len(arr) - 1, dp)
    result = 0
    print(dp)
    for temp in dp:
        result = max(result, temp[0])
    return result

# arr=[20 ,7, 9 ,6, 9 ,21, 12, 3 ,12, 16, 1, 27]
# print(longest_bitonic_subsequence_init(arr))


# Find permutation of Climbing Stair
def find_stair_case(num):
    if num<=1:
        return 1
    count=find_stair_case(num-1)+find_stair_case(num-2)
    return count

# num=5
# print(find_stair_case(num))

def find_stair_case_mem(num,dp):
    if num<=1:
        dp[num]=1
        return 1
    if dp[num]!=-1:
        return dp[num]
    count=0
    if dp[num-1]==-1:
        dp[num-1]=find_stair_case_mem(num-1,dp)
    count+=dp[num-1]
    if dp[num-2]==-1:
        dp[num-2]=find_stair_case_mem(num-2,dp)
    count+=dp[num-2]
    dp[num]=count
    return count

# num=6
# dp=[-1 for i in range(num+1)]
# print(find_stair_case_mem(num,dp))
# print(dp)

def find_stair_case_dp(num):
    dp=[1,1]
    flag=1
    for i in range(2,num+1):
        temp=sum(dp)
        dp[flag]=temp
        flag=1-flag
    return dp[1-flag]
num=10
# print(find_stair_case_dp(num))

def smart_robbery(arr,index,dp):
    if index>=len(arr)-2:
        dp[index]=max(arr[index],arr[-1])
        return dp[index]

    if dp[index]!=-1:
        return dp[index]
    amount1 = 0
    if index+2<len(arr):
        if dp[index+2]==-1:
            smart_robbery(arr, index + 2,dp)
        amount1=arr[index]+dp[index+2]
    if dp[index+1]==-1:
        smart_robbery(arr, index + 1,dp)
    amount2=dp[index+1]
    dp[index]=max(amount1,amount2)
    return dp[index]
# arr=[49,130,124,85,455,257,341,467,377,432,309,445,440,127,324,38,39,119,83,430,42,334,116,140,159,205,431,478,307,174,387,22,246,425,73,271,330,278,74,98,13,487,291,162,137,356,268,156,75,32,53,351,151,442,225,467,431,108,192,8,338,458,288,254,384,446,410,210,259,222,89,423,447,7,31,414,169,401,92,263,156,411,360,125,38,49,484,96,42,103,351,292,337,375]
#
# dp=[-1 for i in range(len(arr))]
# print(smart_robbery(arr,0,dp))
# print(dp)

def smart_robbery_dp(arr):
    n=len(arr)
    if n==1:
        return arr[0]
    temp=max(arr[0],arr[1])
    dp=[arr[0],temp]
    flag=0
    for i in range(2,n):
        temp=max(dp[flag]+arr[i],dp[1-flag])
        dp[flag]=temp
        flag=1-flag
        print(dp)
    return dp[1-flag]
# arr=[49,130,124,85,455,257,341,467,377,432,309,445,440,127,324,38,39,119,83,430,42,334,116,140,159,205,431,478,307,174,387,22,246,425,73,271,330,278,74,98,13,487,291,162,137,356,268,156,75,32,53,351,151,442,225,467,431,108,192,8,338,458,288,254,384,446,410,210,259,222,89,423,447,7,31,414,169,401,92,263,156,411,360,125,38,49,484,96,42,103,351,292,337,375]
# print(len(arr))
# print(smart_robbery_dp(arr))

def find_total_ways(row,col):
    if row==1 or col==1:
        return 1

    return find_total_ways(row-1,col)+find_total_ways(row,col-1)
# row=2
# col=3
# print(find_total_ways(row,col))

def find_total_ways_dp(row,col,dp):
    if row==1 or col==1:
        dp[row][col]=1
        return 1
    if dp[row][col]!=-1:
        return dp[row][col]
    count=0
    if dp[row-1][col]==-1:
        dp[row-1][col]=find_total_ways_dp(row - 1, col,dp)
    count+=dp[row-1][col]
    if dp[row][col-1]==-1:
        dp[row][col-1]=find_total_ways_dp(row,col-1,dp)
    count+=dp[row][col-1]
    dp[row][col]=count
    return count
# row=3
# col=3
# dp=[[-1 for i in range(col+1)]for j in range(row+1)]
# print(find_total_ways_dp(row,col,dp))
# for i in range(len(dp)):
#     print(dp[i])

# Print the length of longest balance Parenthesis
# Think about the following test cases
# 1. s="()(() ans=2
# 2. s="(()())" ans=6
def maxLength( s):
    # code here
    n = len(s)
    stack = deque()
    stack.append(-1)
    ans = 0
    for i in range(n):
        if s[i] == "(":
            stack.append(i)
        else:
            stack.pop()
            if stack:
                temp = i - stack[-1]
                ans = max(temp, ans)
            else:
                stack.append(i)
    return ans

# S="(()())"
# print(maxLength(S))

# Maximum Rectangle area Given an m x n binary matrix filled with 0's and 1's,
# find the largest square containing only 1's and return its area.

# Recursive Solution
def max_square_area(m,n,grid,ans):
    if m<0 or n<0:
        return 0
    temp=min(max_square_area(m,n-1,grid,ans),max_square_area(m-1,n,grid,ans),max_square_area(m-1,n-1,grid,ans))
    if grid[m][n]==1:
        temp+=1
    else:
        temp=0
    # print(temp,m,n)
    ans[0]=max(ans[0],temp)
    return temp

# matrix = [[0, 1, 1, 0, 1],
#           [1, 1, 0, 1, 0],
#           [1, 1, 1, 1, 0],
#           [1, 1, 1, 1, 0],
#           [1, 1, 1, 1, 1],
#           [0, 0, 0, 0, 0]]
# ans=[0]
# max_square_area(len(matrix)-1,len(matrix[0])-1,matrix,ans)
# print(ans[0])

def max_square_area_mem(m,n,grid,ans,dp):
    if m<0 or n<0:
        return 0
    if dp[m][n]!=-1:
        return dp[m][n]
    temp=min(max_square_area_mem(m,n-1,grid,ans,dp),max_square_area_mem(m-1,n,grid,ans,dp),max_square_area_mem(m-1,n-1,grid,ans,dp))
    if grid[m][n]==1:
        temp+=1
        dp[m][n] = temp
    else:
        dp[m][n]=0
    # print(temp,m,n)
    ans[0]=max(ans[0],dp[m][n])

    return dp[m][n]

# matrix = [[0, 1, 1, 0, 1],
#           [0, 1, 1, 1, 0],
#           [1, 1, 1, 1, 0],
#           [1, 1, 1, 1, 0],
#           [1, 1, 1, 1, 1],
#           [0, 0, 0, 0, 0]]
# ans=[0]
# dp=[[-1 for j in range(len(matrix[0]))]for i in range(len(matrix))]
# max_square_area_mem(len(matrix)-1,len(matrix[0])-1,matrix,ans,dp)
# for i in range(len(dp)):
#     print(dp[i])
# print(ans[0])

# Given an array prices[] which denotes the prices of the stocks on different days,
# the task is to find the maximum profit possible after buying and selling the stocks on different days
# using transaction
def buy_and_sell_stock(prices,n):
    max_profit = 0
    i = 0
    valley = prices[0]
    peak = prices[0]
    ans = []
    while i < n:
        while i < n - 1 and prices[i] >= prices[i + 1]:
            i += 1
        valley = i
        while i < n - 1 and prices[i] <= prices[i + 1]:
            i += 1
        peak = i
        # max_profit+=(prices[peak]-prices[valley])
        if peak != valley:
            # print((valley,peak),end="")
            ans.append(list((valley, peak)))
        i += 1
    return ans

# prices=[6764,3645,5181,5893,4542,6753,3996,5483,585,9895,2657,777,1343,4605,261,2225,959,9884,563,4131,6687,7528,6224,436,3333,110,2037,7007,4710,2310,7596,7827,2307,9129,72,3202,2234,4069,5037,2819,3964,7694,9948,5307,8652,6561,7532,9611,6445,8095,94,9484,1975,6319,9920,5308,6429,1958,8668,7491,620,6264,5318,2927,1745,5391,6129,3979,5812,1167,3150,9776,8861,3098,5083,3865,9659,8968,3476,6104,3415,9923,1940,1743,6242,1861,3403,9023,3819]
# print(buy_and_sell_stock(prices,len(prices)))

# Maximum possible using one transaction can be calculated using the following O(n) algorithm
def find_max_stock_profit(arr,n):
    max_profit=0
    curr_min=arr[0]
    left_to_right=[0]*n
    for i in range(0,n):
        if arr[i]<curr_min:
            curr_min=arr[i]
        else:
            max_profit=max(max_profit,arr[i]-curr_min)
        left_to_right[i]=max_profit
    return left_to_right

# prices=[2, 30, 15, 10, 8, 25, 80]
# print(find_max_stock_profit(prices,len(prices)))

def find_max_stock_profit_reverse(arr,n):
    max_profit=0
    curr_max=arr[-1]
    right_to_left=[0]*n
    for i in range(n-1,-1,-1):
        if arr[i]>curr_max:
            curr_max=arr[i]
        else:
            max_profit=max(max_profit,curr_max-arr[i])
        right_to_left[i]=max_profit
    return right_to_left

# prices=[2, 30, 15, 10, 8, 25, 80]
# print(find_max_stock_profit_reverse(prices,len(prices)))

# In daily share trading, a buyer buys shares in the morning and sells them on the same day.
# If the trader is allowed to make at most 2 transactions in a day, whereas the second transaction
# can only start after the first one is complete (Sell->buy->sell->buy). Given stock prices throughout the day,
# find out the maximum profit that a share trader could have made.
def find_max_profit_from_stock(prices,n):
    left_to_right=find_max_stock_profit(prices,n)
    right_to_left=find_max_stock_profit_reverse(prices,n)
    max_profit=0
    for i in range(n):
        max_profit=max(max_profit,left_to_right[i]+right_to_left[i])
    return max_profit
# prices=[2, 30, 15, 10, 8, 25, 80]
# print(find_max_profit_from_stock(prices,len((prices))))


# Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*' where:
#
# '.' Matches any single character.
# '*' Matches zero or more of the preceding element.
# The matching should cover the entire input string (not partial).
def match_string(s,p,i,j):
    if  j ==len(p):
        print(i,j)
        return i==len(s)

    first_match=i<len(s) and p[j] in {s[i],"."}
    # print(first_match,i,j)
    if j+1<len(p) and p[j+1]=="*":
        result1=match_string(s,p,i,j+2)
        result2=first_match and match_string(s,p,i+1,j)
        return result1 or result2
    else:
        return first_match and match_string(s,p,i+1,j+1)

# s="ab"
# p = ".*"
# print(match_string(s,p,0,0))

def match_string_mem(s,p,i,j,dp):
    if  j ==len(p):
        result=i==len(s)
        dp[i][j]=result
        return result
    if dp[i][j]!=-1:
        return dp[i][j]
    first_match=i<len(s) and p[j] in {s[i],"."}
    # print(first_match,i,j)
    if j+1<len(p) and p[j+1]=="*":
        result1=match_string_mem(s,p,i,j+2,dp)
        result2=first_match and match_string_mem(s,p,i+1,j,dp)
        dp[i][j]=result1 or result2
    else:
        dp[i][j]=first_match and match_string_mem(s,p,i+1,j+1,dp)
    return dp[i][j]

# s = "aab"
# p = "c*a*b"
# dp=[[-1 for i in range(len(p)+1)]for j in range(len(s)+1)]
# print(match_string_mem(s,p,0,0,dp))
# for i in range(len(dp)):
#     print(dp[i])

# Given a string, the task is to count all palindrome sub string in a given string.
# Length of palindrome sub string is greater than or equal to 2.

# Recursive Approach
def is_palindrom(i,j,string):
    if j<=i+1:
        return string[i]==string[j]
    if string[i]!=string[j]:
        return False
    return is_palindrom(i+1,j-1,string)

def count_palindromic_substring(i,j,string):
    if i>=j:
        return 0
    if j==i+1:
        if string[i]==string[j]:
            return 1
        else:
            return 0
    count=count_palindromic_substring(i+1,j,string)+count_palindromic_substring(i,j-1,string)-count_palindromic_substring(i+1,j-1,string)
    if is_palindrom(i,j,string):
        count+=1
    # print(i,j,count)
    return count

# string="abbaeae"
# i=0
# j=len(string)-1
# print(count_palindromic_substring(i,j,string))

# Top to bottom (Memoization) Approach
def is_palindrom_mem(i, j, string, dp):
    if j<= i + 1:
        dp[i][j] = string[i] == string[j]
        return dp[i][j]
    if dp[i][j] != -1:
        return dp[i][j]
    if string[i] == string[j]:
        dp[i][j] = is_palindrom_mem(i + 1, j - 1, string, dp)
    else:
        dp[i][j] = False
    return dp[i][j]

def count_palindromic_substring_mem(string,n):
    dp=[[-1 for i in range(n+1)]for j in range(n+1)]
    count=0
    for i in range(n):
        for j in range(i+1,n):
            if is_palindrom_mem(i,j,string,dp):
                count+=1
    for i in range(len(dp)):
        print(dp[i])
    return count

# string="abbaeae"
# print(count_palindromic_substring_mem(string,len(string)))


# Given a value N, find the number of ways to make change for N cents,
# if we have infinite supply of each of S = { S1, S2, .. , SM } valued coins.
def coin_change(amount,i,coins,dp):
    if amount==0:
        return 1
    if amount<0 or i <0:
        return 0
    if dp[i][amount]!=-1:
        return dp[i][amount]
    count = (coin_change(amount, i - 1, coins, dp)+coin_change(amount-coins[i],i,coins,dp))
    dp[i][amount]=count
    return count

# coins=[2, 5, 3, 6]
# amount=10
# dp=[[-1 for j in range(amount+1)]for i in range(len(coins)+1)]
# print(coin_change(amount,len(coins)-1,coins,dp))
# for i in range(len(dp)):
#     print(dp[i])

def coin_change_dp(coins,amount):
    dp = [0 for j in range(amount + 1)]
    dp[0]=1
    for i in range(len(coins)):
        for j in range(1,amount+1):
            if j-coins[i]>=0:
                dp[j]+=dp[j-coins[i]]
        print(dp)
    return dp[amount]
# coins=[2, 5, 3, 6]
# amount=10
# print(coin_change_dp(coins,amount))

def max_cost_path(grid,i,j,dp):
    # print(i, j)
    # if i==len(grid)-1 and j<=len(grid[0])-1:
    #     dp[i][j]=grid[i][j]
    #     return dp[i][j]
    if i>=len(grid) or j>=len(grid[0]) or j<0:
        return 0
    if dp[i][j]!=-1:
        return dp[i][j]

    cost1=max_cost_path(grid,i+1,j-1,dp)
    cost2=max_cost_path(grid,i+1,j,dp)
    cost3=max_cost_path(grid,i+1,j+1,dp)
    dp[i][j]=grid[i][j]+max(cost1,cost2,cost3)
    return dp[i][j]

def max_cost_path_init(cost):
    dp = [[-1 for j in range(len(cost[0]))] for i in range(len(cost))]
    ans = 0
    for i in range(len(cost[0])):
        ans = max(ans, max_cost_path(cost, 0, i, dp))
    for i in range(len(dp)):
        print(dp[i])
    return ans

cost = [[1, 0, 7],
        [2, 0, 6],
        [3, 4, 5],
        [0, 3, 0],
        [9, 0, 20]]
# print(max_cost_path_init(cost))

# Gold Mine Problem
# Given a gold mine called M of (n x m) dimensions. Each field in this mine contains a positive integer which
# is the amount of gold in tons. Initially the miner can start from any row in the first column.
# From a given cell, the miner can move
#
# to the cell diagonally up towards the right
# to the right
# to the cell diagonally down towards the right
# Find out maximum amount of gold which he can collect.
def mine_for_gold(grid,i,j,dp):
    if i>=len(grid) or i<0 or j>=len(grid[0]) or j<0:
        return 0
    if dp[i][j]!=-1:
        return dp[i][j]
    cost1=mine_for_gold(grid,i-1,j+1,dp)
    cost2=mine_for_gold(grid,i,j+1,dp)
    cost3=mine_for_gold(grid,i+1,j+1,dp)
    dp[i][j]=grid[i][j]+max(cost1,cost2,cost3)
    # print(i,j)
    return dp[i][j]

# grid = [[10, 33, 13, 15],
#         [22, 21, 4, 1],
#         [5, 0, 2, 3],
#         [0, 6, 14, 2]]
# ans=0
# dp=[[-1 for j in range(len(grid[0]))]for i in range(len(grid))]
# for i in range(len(grid)):
#     ans=max(ans,mine_for_gold(grid,i,0,dp))
# print(ans)
# for i in range(len(dp)):
#     print(dp[i])

# A number can always be represented as a sum of squares of other numbers.
# Note that 1 is a square and we can always break a number as (1*1 + 1*1 + 1*1 + …).
# Given a number n, find the minimum number of squares that sum to X.
def min_square_sum(n,dp):
    # print(n)
    if n<0:
        return inf
    if n<=3:
        return n
    if dp[n]!=-1:
        return dp[n]
    ans=inf
    temp=int(n**0.5)
    # print(n,temp)
    for i in range(1,temp+1):
        num=n-(i*i)
        # print(num)
        ans=min(ans,min_square_sum(num,dp))
    ans += 1
    dp[n]=ans
    return ans
# n = 100
# dp=[-1 for i in range(n+1)]
# print(min_square_sum(n,dp))
# print(dp)


# Targeted Sum
# You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -.
# For each integer, you should choose one from + and - as its new symbol.
#
# Find out how many ways to assign symbols to make sum of integers equal to target S.
def find_all_combinations(i,s1,s2,num,arr,dp):
    if  i==len(arr):
        if s1-s2==num:
            return 1
        else:
            return 0
    if dp[s1][s2]!=-1:
        return dp[s1][s2]

    count=find_all_combinations(i+1,s1+arr[i],s2,num,arr,dp)
    count+=find_all_combinations(i+1,s1,s2+arr[i],num,arr,dp)
    dp[s1][s2]=count
    return count

# arr=[3, 2, 2, 2, 4]
# num=3
# s=sum(arr)
# dp=[[-1 for j in range(s+1)]for i in range(s+1)]
# print(find_all_combinations(0,0,0,num,arr,dp))
# for i in range(len(dp)):
#     print(dp[i])

# Ugly numbers are numbers whose only prime factors are 2, 3 or 5.
# The sequence 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, … shows the first 11 ugly numbers.
# By convention, 1 is included.
# Given a number n, the task is to find n’th Ugly number.
def getNthUglyNo(n):
    # code here
    arr=[0]*n
    i2=i3=i5=0
    arr[0]=1
    next_num_2=2
    next_num_3=3
    next_num_5=5
    for i in range(1,n):
        next_num=min(next_num_2,next_num_3,next_num_5)
        if next_num==next_num_2:
            arr[i]=next_num
            i2+=1
            next_num_2=arr[i2]*2
        if next_num==next_num_3:
            arr[i]=next_num
            i3 += 1
            next_num_3=arr[i3]*3
        if next_num==next_num_5:
            arr[i]=next_num
            i5 += 1
            next_num_5=arr[i5]*5
    print(arr)
    return arr[-1]
# n=10
# print(getNthUglyNo(150))


def max_square(grid,i,j,dp):
    if i<0 or j<0:
        return inf
    if i==0 or j==0:
        if grid[i][j]==1:
            dp[i][j]=1
            return 1
    if dp[i][j]!=-1:
        return dp[i][j]
    area=min(max_square(grid,i-1,j,dp),max_square(grid,i,j-1,dp),max_square(grid,i-1,j-1,dp))
    if grid[i][j]==1:
        area+=1
    else:
        area=0
    dp[i][j]=area
    # print(i, j,area)
    return area
# grid = [[0, 1, 1, 0, 1],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0]]

# dp=[[-1 for j in range(len(grid[0]))]for i in range(len(grid))]
# max_square(grid,len(grid)-1,len(grid[0])-1,dp)
# # print(max(dp))
# max_area=0
# for i in range(len(dp)):
#     max_area=max(max_area,max(dp[i]))
# print(max_area)

def partition_the_arr(s1,s2,arr,i,dp):
    if i==len(arr):
        if s1==s2:
            return True
        else:
            return False
    diff=abs(s1-s2)
    if dp[i][diff]!=-1:
        return dp[i][diff]
    dp[i][diff]=partition_the_arr(s1+arr[i],s2,arr,i+1,dp) or partition_the_arr(s1,s2+arr[i],arr,i+1,dp)
    return dp[i][diff]

# arr=[1, 5, 11, 5]
# dp=[[-1 for j in range(sum(arr))]for i in range(len(arr))]
# print(partition_the_arr(0,0,arr,0,dp))

def partition_the_arr_dp(arr):
    s=sum(arr)
    if s%2!=0:
        return False

    dp = [[False for j in range(s//2+1)] for i in range(len(arr))]
    dp[0][arr[0]]=True
    temp=arr[0]
    for i in range(1,len(arr)):
        temp+=arr[i]
        for j in range(min(temp+1,s//2+1)):
            dp[i][j]=dp[i-1][j]
            if j-arr[i]>=0:
                dp[i][j]=dp[i][j] or dp[i-1][j-arr[i]]
    for i in range(len(dp)):
        print(dp[i])
    print(dp[len(arr)-1][-1])
# arr=[1, 5, 11, 5]
# print(partition_the_arr_dp(arr))
# print(sum(arr))


# Given an array of integers where each element represents the max number of steps that can be made
# forward from that element. Write a function to return the minimum number of jumps to reach the end of the array
# (starting from the first element). If an element is 0, they cannot move through that element.
# If the end isn’t reachable, return -1.
def find_min_jump(arr,i,dp):
    if i>=len(arr):
        return inf
    if i==len(arr)-1:
        return 0
    if dp[i]!=-1:
        return dp[i]
    temp=arr[i]
    min_jump=inf
    for j in range(i+1,min(len(arr),i+temp+1)):
        min_jump=min(min_jump,find_min_jump(arr,j,dp))
    dp[i]=min_jump+1
    return dp[i]

# arr=[1, 3, 6, 3, 2, 3, 6, 8, 9, 5]
# dp=[-1 for i in range(len(arr))]
# print(find_min_jump(arr,0,dp))
# print(dp)

# Optimization of TC=O(N) and S.C=O(N)
def minJumps( arr, n):
    start,end,step=0,0,0
    while end<n:
        step+=1
        max_end=end
        while start<=end:
            max_end=max(max_end,start+arr[start])
            if max_end>=n-1:
                return step
            start+=1
        end=max_end
    return step

# arr=[1,3,5,8,9,2,6,7,6,8,9,]
# print(minJumps(arr,len(arr)))

def find_max_bitonic_sum(arr,n):
    left=[0]*n
    for i in range(n):
        temp=0
        for j in range(0,i):
            if arr[j]<arr[i]:
                temp=max(temp,left[j])
        left[i]=temp+arr[i]
    right=[0]*n
    for i in range(n-1,-1,-1):
        temp=0
        for j in range(i+1,n):
            if arr[i]>arr[j]:
                temp=max(temp,right[j])
        right[i]=temp+arr[i]
    max_sum=0
    for i in range(n):
        max_sum=max(max_sum,left[i]+right[i]-arr[i])
    return max_sum
# arr=[80, 60, 30, 40, 20, 10]
# print(find_max_bitonic_sum(arr,len(arr)))

#Given 3 strings X, Y and Z, the task is to find the longest common sub-sequence in all three given sequences.
def find_lcs(s1,s2,s3,i,j,k,dp):
    if i==len(s1) or j==len(s2) or k==len(s3):
        return 0
    if dp[i][j][k]!=-1:
        return dp[i][j][k]
    if s1[i]==s2[j]==s3[k]:
        return 1+find_lcs(s1,s2,s3,i+1,j+1,k+1,dp)
    result1 = find_lcs(s1, s2, s3, i+1, j, k,dp)
    result2 = find_lcs(s1, s2, s3, i, j+1, k,dp)
    result3 = find_lcs(s1, s2, s3, i, j, k+1,dp)
    dp[i][j][k]=max(result1,result2,result3)
    return dp[i][j][k]
# X = 'geeks'
# Y = 'geeksfor'
# Z = 'geeksforgeeks'
#
# dp=[[[-1 for k in range(len(Z))]for j in range(len(Y))]for i in range(len(X))]
#
# print(find_lcs(X,Y,Z,0,0,0,dp))

def find_lcs_dp(s1,s2,s3,a,b,c):
    dp = [[[-1 for k in range(c+1)] for j in range(b+1)] for i in range(a+1)]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if i==0 or j==0 or k==0:
                    dp[0][j][k]=0

    for i in range(a):
        for j in range(b):
            for k in range(c):
                if s1[i] == s2[j] == s3[k]:
                    dp[i+1][j+1][k+1]= 1 +dp[i][j][k]
                else:
                    dp[i+1][j+1][k+1]=max(dp[i][j+1][k+1],dp[i+1][j][k+1],dp[i+1][j+1][k])
    return dp[a][b][c]

# X = 'lkbkde'
# Y = 'cdelkb'
# Z = 'abcd'
#
# print(find_lcs_dp(X,Y,Z,len(X),len(Y),len(Z)))

# Given n friends, each one can remain single or can be paired up with some other friend.
# Each friend can be paired only once. Find out the total number of ways in which friends can
# remain single or can be paired up.

def find_friends_pair(n,dp):
    if n<=2:
        dp[n]=n
        return n
    if dp[n]!=-1:
        return dp[n]
    dp[n]=find_friends_pair(n-1,dp) + (n-1)*find_friends_pair(n-2,dp)
    return dp[n]

# n=3
# dp=[-1 for i in range(n+1)]
# print(find_friends_pair(n,dp))
# print(dp)

def find_friends_pair_dp(n):
    dp=[0,1,2]
    for i in range(3,n+1):
        dp[i%3]=dp[(i-1)%3]+ (i-1)*dp[(i-2)%3]
    return dp[n%3]
# n=5
# print(find_friends_pair_dp(n))

# Building Bridges
# https://www.geeksforgeeks.org/dynamic-programming-building-bridges/
def sorted_fun(x,y):
    if x[0]==y[0]:
        return x[1]<y[1]
    return x[0]<y[0]
def build_max_bridge(bridges,n):
    dp=[-1 for i in range(n)]
    bridges=sorted(bridges,key=itemgetter(0,1))
    dp[0]=1
    for i in range(1,n):
        curr_max=0
        for j in range(0,i):
            if bridges[i][1]>bridges[j][1]:
                curr_max=max(curr_max,dp[j])
        dp[i]=curr_max+1
    max_bridges=0
    for i in range(n):
        max_bridges=max(max_bridges,dp[i])
    return max_bridges

# bridges=[[6, 2], [4, 3], [2, 6], [1, 5]]
# print(build_max_bridge(bridges,len(bridges)))

def partition_sum(arr,i,sum,dp):
    if i<0 or sum <0:
        return False
    if sum==0:
        dp[i][sum]=True
        return True
    if dp[i][sum]!=-1:
        return dp[i][sum]
    dp[i][sum]=partition_sum(arr,i-1,sum,dp) or partition_sum(arr,i,sum-arr[i],dp)
    return dp[i][sum]

# arr=[1, 3, 5]
# s=sum(arr)
# if s%2!=0:
#     print(False)
# else:
#     dp = [[-1 for j in range((s // 2)+1)] for i in range(len(arr))]
#     print(partition_sum(arr,len(arr)-1,s//2,dp))


# Given two numbers n and k where n represents a number of elements in a set,
# find a number of ways to partition the set into k subsets.
def partition_into_k_subset(n,k,dp):
    if k>n or n<=0:
        return 0
    if k==1:
        return 1
    if dp[k][n]!=-1:
        return dp[k][n]

    dp[k][n]=k*partition_into_k_subset(n-1,k,dp)+ partition_into_k_subset(n-1,k-1,dp)
    return dp[k][n]
# n=5
# k=2
# dp=[[-1 for j in range(n+1)]for i in range(k+1)]
# print(partition_into_k_subset(n,k,dp))

# Given a sequence, find the length of the longest palindromic subsequence in it.
def longest_palindrom_subseq(string,i,j,dp):
    if j==i:
        return 1
    if i-j==1:
        dp[i][j]=2 if string[i]==string[j] else 1
        return dp[i][j]
    if dp[i][j]!=-1:
        return dp[i][j]
    result=0
    if string[i]==string[j]:
        result=longest_palindrom_subseq(string,i-1,j+1,dp)+2
    else:
        ans2 = longest_palindrom_subseq(string, i, j + 1,dp)
        ans3 = longest_palindrom_subseq(string, i - 1, j,dp)
        result=max(ans3,ans2)
    dp[i][j]=result
    return result
# string="BBABCBCAB"
# dp=[[-1 for i in range(len(string))]for j in range(len(string))]
# i=len(string)-1
# print(longest_palindrom_subseq(string,i,0,dp))
# for i in range(len(string)):
#     print(dp[i])

# Egg Dropping Puzzle
def find_min_attemps(f,eggs,dp):
    if eggs==1:
        return f
    if f==1 or f==0:
        return f
    # print(f, eggs)
    if dp[eggs][f]!=-1:
        return dp[eggs][f]
    attemps=inf
    for i in range(1,f):
        temp=max(find_min_attemps(i-1,eggs-1,dp),find_min_attemps(f-i,eggs,dp))+1
        # print(temp,i,f)
        attemps=min(attemps,temp)
    dp[eggs][f]=attemps
    return dp[eggs][f]

# f=16
# eggs=8
# dp=[[-1 for j in range(f+1)]for i in range(eggs+1)]
# print(find_min_attemps(f,eggs,dp))



#
class Pair(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

def maxChainLen(arr, n):
    # Parr:  list of pair
    # code here
    arr=sorted(arr,key=attrgetter("a"))
    for i in range(len(arr)):
        print(arr[i].a,arr[i].b,end=" ")
    dp = [0 for k in range(len(Parr))]
    for i in range(len(dp)):
        temp=0
        for  j in range(i):
            if arr[j].b < arr[i].a:
                temp = max(temp, dp[j])
        dp[i]=temp+1
    ans=0
    print()
    # print(dp)
    for i in range(len(dp)):
        ans=max(ans,dp[i])
    return ans

# arr = [5,24,39,60,15,28,27,40,50,90]
# n=len(arr)//2
# Parr = []
# i = 0
# while n * 2 > i:
#     Parr.append(Pair(arr[i], arr[i + 1]))
#     i += 2
# # print(Parr,len(Parr))
# print(maxChainLen(Parr, n))
#Box Stacking Problem

def box_stacking(height,width,length,n):
    boxes=[]
    for i in range(n):
        box1=[height[i],max(width[i],length[i]),min(width[i],length[i])]
        box2=[width[i],max(height[i],length[i]),min(height[i],length[i])]
        box3=[length[i],max(height[i],width[i]),min(height[i],width[i])]
        boxes.append(box1)
        boxes.append(box2)
        boxes.append(box3)
    boxes=sorted(boxes,reverse=True,key=lambda x:(x[1]*x[2]))
    size=len(boxes)
    # print(boxes)
    dp=[-1 for i in range(size)]
    for i in range(len(boxes)):
        h=0
        for j in range(i):
            if boxes[j][2]>boxes[i][2] and boxes[j][1]>boxes[i][1]:
                h=max(h,dp[j])
        dp[i]=h+boxes[i][0]
    max_height=0
    # print(dp)
    for i in range(size):
        max_height=max(max_height,dp[i])
    return max_height

# height = [1,4,3]
# width= [2,5,4]
# length = [3,6,1]
# n=len(length)
# print(box_stacking(height,width,length,n))

# Matrix Chain Multiplication
def min_operations(arr,j,i,dp):

    if i==j:
        return 0
    if i-j==1:
        dp[i][j]=arr[j]*arr[i+1]*arr[i]
        return dp[i][j]
    if dp[i][j]!=-1:
        return dp[i][j]
    operations=inf
    for  k in range(j,i):
        operations=min(operations,min_operations(arr, j,k, dp) +
                       min_operations(arr,k+1,i,dp)+ arr[j]*arr[k+1]*arr[i+1])
    dp[i][j]=operations
    # print(j, i,operations)
    return operations

# arr=[1, 2, 3, 4, 3]
# dp=[[-1 for i in range(len(arr))]for j in range(len(arr))]
# print(min_operations(arr,0,len(arr)-2,dp))
# for i in range(len(dp)):
#     print(dp[i])

# Optimal Binary Search Tree
def min_search_cost(start,end,freq,dp):
    if 0>start or end>=len(freq) or start>end:
        return 0
    if start==end:
        return freq[start]
    if dp[start][end]!=-1:
        return dp[start][end]
    min_cost=inf
    curr_sum=sum(freq[start:end+1])
    for i in range(start,end+1):
        left=min_search_cost(start,i-1,freq,dp)
        right=min_search_cost(i+1,end,freq,dp)
        min_cost=min(min_cost,left+curr_sum+right)
        dp[start][end]=min_cost
    return min_cost

# freq=[2,3,4]
# dp=[[-1 for j in range(len(freq))]for i in range(len(freq))]
# print(min_search_cost(0,len(freq)-1,freq,dp))
# for i in range(len(dp)):
#     print(dp[i])


# Minimum insertions to form a palindrome
# Given a string str, the task is to find the minimum number of characters to be inserted
# to convert it to palindrome.
# Before we go further, let us understand with few examples:
def min_insertion(string,start,end,dp):
    if start==end:
        return 0
    if end-start==1:
        return 0 if string[start]==string[end] else 1
    if dp[start][end]!=-1:
        return dp[start][end]
    if string[start]==string[end]:
        return min_insertion(string,start+1,end-1,dp)
    call1=min_insertion(string,start+1,end,dp)+1
    call2=min_insertion(string,start,end-1,dp)+1
    dp[start][end]=min(call1,call2)
    # print(start,end,)
    return dp[start][end]

# string="pankaj"
# dp=[[-1 for i in range(len(string))]for j in range(len(string))]
# print(min_insertion(string,0,len(string)-1,dp))
# for i in range(len(dp)):
#     print(dp[i])


#Maximum Product Cutting
# Given a rope of length n meters, cut the rope in different parts of integer
# lengths in a way that maximizes product of lengths of all parts.
# You must make at least one cut. Assume that the length of rope is more than 2 meters.
def max_rope_product(n,dp):
    if n==1:
        return 1
    if n<=3:
        return n-1
    if dp[n]!=-1:
        return dp[n]
    product=0
    for i in range(1,n//2+1):
        temp=max(i,max_rope_product(i,dp))*max(n-i,max_rope_product(n-i,dp))
        product=max(product,temp)
    dp[n]=product
    return product
# n=57
# dp=[-1 for i in range(n+1)]
# print(max_rope_product(n,dp))
# print(dp)

# Cutting a Rod
# Given a rod of length n inches and an array of prices that contains prices of all pieces of size smaller than n.
# Determine the maximum value obtainable by cutting up the rod and selling the pieces.
def max_profit(length,arr,dp):
    # print(length)
    if dp[length]!=-1:
        return dp[length]
    curr_profit=0
    for i in range(1,length//2+1):
        curr_profit=max(curr_profit,max_profit(i,arr,dp)+max_profit(length-i,arr,dp))
    dp[length]=max(arr[length-1],curr_profit)
    return curr_profit

# Driver code
# arr = [3, 5, 8, 9, 10, 17, 17, 20]
# size = len(arr)
# dp=[-1 for i in range(size+1)]
# print(max_profit(size,arr,dp))
# print(dp)

#Given an integer N denoting the Length of a line segment. You need to cut the line segment
# in such a way that the cut length of a line segment each time is either x , y or z.
# Here x, y, and z are integers.
# After performing all the cut operations, your total number of cut segments must be maximum.
def max_segment(n,x,y,z,dp):
    if n<x:
        return 0
    if n%x==0:
        dp[n]=n//x
        return dp[n]
    if dp[n]!=-1:
        return dp[n]
    result1=max_segment(n-z,x,y,z,dp)
    if result1!=0:
        result1+=1
    result2=max_segment(n-y,x,y,z,dp)
    if result2!=0:
        result2+=1
    dp[n]=max(result1,result2)
    return dp[n]

# N = 17
# x = 5
# y = 3
# z = 2
# x,y,z=sorted((x,y,z))
# dp=[-1 for i in range(N+1)]
# print(max_segment(N,x,y,z,dp))
# print(dp)









































