
#Longest Increasing Subsequence Size (N log N)
# https://www.geeksforgeeks.org/longest-monotonically-increasing-subsequence-size-n-log-n/
# https://youtu.be/66w10xKzbRM
def find_index(arr,num,start,end):
    if start>=end:
        return start
    mid=(start+end)//2
    if arr[mid]<num:
        start=mid+1
    elif arr[mid]>num:
        end=mid
    else:
        return mid
    return find_index(arr,num,start,end)

def longestSubsequence(arr,n):
    # code here
    # return length of the longest increasing sub sequence
    dp=[0]*(n+1)
    dp[0]=arr[0]
    size=1
    for i in range(1,n):
        if arr[i]>dp[size-1]:
            dp[size]=arr[i]
            size+=1
        else:
            index=find_index(dp,arr[i],0,size)
            print(arr[i],index)
            dp[index]=arr[i]
    print(dp)
    return size


# arr=[6 ,3 ,7 ,4, 6, 9]
# print(longestSubsequence(arr,len(arr)))
# print(find_index(sorted(arr),4,0,len(arr)))

# Longest Increasing Bitonic Subsequence

def longest_bitonic_subsequence(arr,n):
    left=[-1]*(n)
    left[0]=1
    for i in range(n):
        temp=0
        for j in range(0,i):
            if arr[i]>arr[j]:
                temp=max(temp,left[j])
        left[i]=temp+1
    right=[-1]*(n)
    right[n-1]=1
    for i in range(n-2,-1,-1):
        temp=0
        for j in range(i,n):
            if arr[i]>arr[j]:
                temp=max(temp,right[j])
        right[i]=temp+1
    ans=0
    for i in range(n):
        ans=max(left[i]+right[i]-1,ans)
    return ans

arr=[0 , 8 , 4, 12, 2, 10 , 6 , 14 , 1 , 9 , 5 , 13, 3, 11 , 7 , 15]
print(longest_bitonic_subsequence(arr,len(arr)))















