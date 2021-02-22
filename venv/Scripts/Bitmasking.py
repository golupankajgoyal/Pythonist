import sys

def minCostJob(costs,n,p,mask,dp):
    if p==n:
        return 0
    if dp[mask]!=0:
        return dp[mask]
    minimum=sys.maxsize
    for i in range(n):
        if not(mask&(1<<i)):
            ans=minCostJob(costs,n,p+1,(mask|(1<<i)),dp)+costs[p][i]
            if ans<minimum:
                minimum=ans
    dp[mask]=minimum
    return dp[mask]

def minCostJobdp(costs,n):
    dp=[sys.maxsize]*((1<<n))
    dp[0]=0
    for mask in range(1<<n):
        temp=mask
        setBit=0
        while temp>0:
            if temp&(1):
                setBit+=1
            temp=temp>>1
        for j in range(n):
            if not(mask&(1<<j)):
                dp[mask|1<<j]=min(dp[mask|1<<j],dp[mask]+costs[setBit][j])
    print(dp[((1<<n)-1)])
    print(dp)

def uniqueStrings(mask,pos,strings,n,dp):
    print(mask,pos)
    if mask&(mask-1)==0:
        return 0
    if pos>=n:
        return 1000
    if dp[pos][mask]!=-1:
        return dp[pos][mask]
    temp=mask
    mask1=mask2=0
    setBit=count1=count2=0
    # while temp>0:
    #     setBit+=(temp&1)
    #     temp=temp//2
    for i in range(len(strings)):
        if mask&(1<<i):
            if strings[i][pos]==1:
                setBit+=1
                mask1|=1<<i
            else:
                mask2|=1<<i
    count1=uniqueStrings(mask1,pos+1,strings,n,dp)
    count2=uniqueStrings(mask2,pos+1,strings,n,dp)
    ans=min(setBit+count1+count2,uniqueStrings(mask,pos+1,strings,n,dp))
    dp[pos][mask]=ans
    return ans


strings = [[1, 1, 1, 0, 1, 0],
           [1, 0, 0, 1, 0, 0],
           [1, 1, 0, 1, 0, 0],
           [1, 1, 1, 1, 0, 0]]
dp=[[-1 for j in range(1<<len(strings))]for i in range(len(strings[0]))]
print(uniqueStrings((1<<len(strings))-1,0,strings,len(strings[0]),dp))
for i in range(len(dp)):
    print(dp[i])
