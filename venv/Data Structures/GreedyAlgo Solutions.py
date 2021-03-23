class Job:
    def __init__(self,start=0,finish=0,price=0):
        self.start=start
        self.finish=finish
        self.price=price

def compare(j1):
    return j1.finish


def findPrevPossibleJob(arr, start,end,value):
    if end-start<=1:
        return start
    mid=(start+end)//2
    if arr[mid].finish<value:
        start=mid
    else:
        end=mid
    return findPrevPossibleJob(arr,start,end,value)


def weightedJobScheduling(arr,index,dp):
    if index==0:
        dp[0]=arr[0].price
        return dp[0]
    if dp[index]!=-1:
        return dp[index]
    prevJob=weightedJobScheduling(arr,index-1,dp)
    suitableJob=findPrevPossibleJob(arr,0,index,arr[index-1].start)
    dp[index-1]=max(arr[index-1].price+arr[suitableJob].price,prevJob)
    return dp[index-1]



j1=Job(2,5,50)
j2=Job(1, 2, 20)
j3=Job(6, 9, 70)
j4=Job(2, 50, 400)
arr=[j1,j2,j3,j4]
arr=sorted(arr,key=compare)
dp=[-1]*(len(arr)+1)
print(weightedJobScheduling(arr,len(arr),dp))
