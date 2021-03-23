import heapq

def findKlargeElements(arr, k):
    size=len(arr)
    for i in range(size):
        arr[i]=-arr[i]
    heapq.heapify(arr)
    ans=[]
    for _ in range(k):
        ans.append(-heapq.heappop(arr))
    return ans

def findKMax(arr,n,k):
    heap=arr[:k]
    heapq.heapify(heap)
    for i in arr[k:]:
        if heap[0]<i:
            heapq.heapreplace(heap,i)
    print(heap[0])

def findSmallestInteger(arr,n):
    endIndex=partition(arr,n)
    print(arr)
    for i in range(endIndex+1):
        curr=abs(arr[i])-1
        if curr <= endIndex:
            arr[curr]=-abs(arr[curr])
    for i in range(endIndex+1):
        if arr[i]>0:
            return i+1

def partition(arr,n):
    i=0
    j=n-1
    while i<j:
        if arr[j]<0:
            j-=1
        elif arr[i]<0:
            arr[i],arr[j]=arr[j],arr[i]
        else:
            i+=1
    return i

# Design a class to calculate the median of a number stream. The class should have the following two methods:
#
# insertNum(int num): stores the number in the class
# findMedian(): returns the median of all numbers inserted in the class
# If the count of numbers inserted in the class is even, the median will be the average of the middle two numbers.
class MeadianOfAStream:
    def __init__(self):
        self.min_heap=[]
        self.max_heap=[]

    def insert(self,num):

        if len(self.max_heap)==0 or -self.max_heap[0]<num:
            heapq.heappush(self.max_heap,-num)
        else:
            heapq.heappush(self.min_heap,num)

        if len(self.max_heap)>len(self.min_heap)+1:
            heapq.heappush(self.min_heap,-heapq.heappop(self.max_heap))
        elif len(self.min_heap)>len(self.max_heap):
            heapq.heappush(self.max_heap,-heapq.heappop(self.min_heap))

    def find_median(self):
        if len(self.min_heap)==len(self.max_heap):
            return -self.max_heap[0]/2 + self.min_heap[0]/2
        return -self.max_heap[0]/1
