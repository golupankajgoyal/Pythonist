from collections import deque
from math import inf
def findDoubleInExpression(string):
    stack=deque()
    for char in string:
        if char is ")":
            curr=stack.pop()
            if curr is "(" :
                print("Found Duplicate")
                return
            else:
                while curr is not "(":
                    curr=stack.pop()
        else:
            stack.append(char)
    print("No Double")

def sumOfMinMaxOfKSizeWindow(arr,n,k):
    sum=0
    maxq=deque()
    minq=deque()
    for i in range(n):
        if len(maxq)>0 and maxq[0]<i-k+1:
            maxq.popleft()
        if len(minq)>0 and maxq[0]<i-k+1:
            minq.popleft()
        while len(maxq)>0 and arr[maxq[-1]]<arr[i]:
            maxq.pop()
        while len(minq)>0 and arr[minq[-1]]>arr[i]:
            minq.pop()
        maxq.append(i)
        minq.append(i)
        if i>=k-1:
            sum+=(arr[minq[0]]+arr[maxq[0]])
    return sum
# //////////////////////////////////////////////////////////////
# below two functions have same functionality
def findMaxAreaRectangle(arr,size):
    s=deque()
    width=[0]*size
    for i in range(size-1,-1,-1):
        while len(s)>0 and arr[s[-1]]>=arr[i]:
            s.pop()
        if len(s)==0:
            width[i]=size
        else:
            width[i]=s[-1]
        s.append(i)
    s=deque()
    for i in range(size):
        while len(s)>0 and arr[s[-1]]>=arr[i]:
            s.pop()
        if len(s)!=0:
            width[i]-=(s[-1]+1)
        s.append(i)
    result=0
    for i in range(size):
        result=max(result,width[i]*arr[i])
    print(result)

def max_area_histogram(arr,size):
    s=deque()
    i=0
    result=0
    while i<size :
        if len(s) is 0 or arr[s[-1]]<=arr[i]:
            s.append(i)
            i+=1
        else:
            area=arr[s.pop()]*((i - s[-1]-1)if len(s)>0 else i)
            result=max(result,area)
    while len(s)>0:
        area=arr[s.pop()]*((i - s[-1]-1)if len(s)>0 else i)
        result=max(result,area)
    return result

# /////////////////////////////////////////////////////////////////////////////
def maxInSizeK(arr):
    for j in range(1,len(arr)+1):
        d=deque()
        currMax=-1
        for i in range(len(arr)):
            if len(d)>0 and d[0]<i-j+1:
                d.popleft()
            while len(d)>0 and arr[d[-1]]>arr[i]:
                d.pop()
            d.append(i)
            if i>j-2:
                currMax=max(arr[d[0]],currMax)
        print(currMax,end=" ")

# /////////////////////////////////////////////////////////////////////////////

def nextLargerElement(arr,n):
    stack=deque()
    stack.append(-1)
    ans=[-1 for  i in range(n)]
    for i in range(n-1,-1,-1):
        if stack:
            top=stack.pop()
            if top < arr[i]:
                while stack and top < arr[i]:
                    top=stack.pop()
            ans[i]=top
            stack.append(top)
        stack.append(arr[i])
    return ans
# n=4
# arr=[3,1,4,2]
# print(nextLargerElement(arr,n))





















# def find_height(arr,n):
#     ans=0
#     pos=[0,0]
#     for i in range(n):
#         for j in range(i+1,n):
#             if arr[j]>=arr[i] and j-i>ans:
#                 ans=j-i
#                 pos=[i,j]
#     print(ans)
#     print(pos[0],pos[1])
# n=int(input())
# arr=list(map(int,input().split()))
# find_height(arr,n)



