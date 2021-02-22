import sys
from sys import maxsize
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
