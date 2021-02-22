
# We are given an array containing ‘n’ distinct numbers taken from the range 0 to ‘n’.
# Since the array has only ‘n’ numbers out of the total ‘n+1’ numbers, find the missing number.
def cyclicSort(arr):
    i=0
    n=len(arr)
    while i < n:
        j=arr[i]
        if arr[i]<n and arr[i]!=arr[j]:
            arr[j],arr[i]=arr[i],arr[j]
        else:
            i+=1
        print(arr)
    for i in range(len(arr)):
        if i!=arr[i]:
            print(i)
            return

# arr=[8, 7, 5, 2, 4, 6, 0, 1]
# cyclicSort(arr)

def findAllMissing(arr):
    i=0
    n=len(arr)
    while i < n:
        j=arr[i]-1
        if arr[i]!=arr[j]:
            arr[j],arr[i]=arr[i],arr[j]
        else:
            i+=1
    print(arr)
    for i in range(len(arr)):
        if i!=arr[i]-1:
            print(i+1,end=" ")
# arr=[2, 3, 2, 1]
# findAllMissing(arr)

def findDuplicate(arr):
    i=0
    n=len(arr)
    while i < n:
        if arr[i]!=i:
            j=arr[i]
            if arr[i]!=arr[j]:
                arr[j],arr[i]=arr[i],arr[j]
            else:
                print(arr[i])
                return
        else:
            i+=1

# arr=[1, 4, 4, 3, 2]
# findDuplicate(arr)

def findAllDuplicate(arr):
    i=0
    n=len(arr)
    while i < n:
        j=arr[i]-1
        if j<n and arr[i]!=arr[j]:
            arr[j],arr[i]=arr[i],arr[j]
        else:
            i+=1
    for i in range(n):
        if i !=arr[i]-1:
            print(arr[i],end=" ")
# arr=[5, 4, 7, 2, 3, 5, 3]
# findAllDuplicate(arr)

def findCorruptPair(arr):
    i=0
    n=len(arr)
    while i < n:
        j=arr[i]-1
        if j<n and arr[i]!=arr[j]:
            arr[j],arr[i]=arr[i],arr[j]
        else:
            i+=1
    for i in range(n):
        if i !=arr[i]-1:
            print(arr[i],i+1)
            return
# arr=[3, 1, 2, 3, 6, 4]
# findCorruptPair(arr)

# Given an unsorted array containing numbers, find the smallest missing positive number in it.
def smallestMissing(arr):
    i=0
    n=len(arr)
    while i<n:
        j=arr[i]-1
        if 0<=j<n and arr[i]!=arr[j]:
            arr[j],arr[i]=arr[i],arr[j]
        else:
            i+=1
    for i in range(1,n):
        if i !=arr[i]-1:
            print(i+1)
            return
    print(n+1)
# arr=[-3, 1, 5, 4, 2]
# smallestMissing(arr)

#Given an unsorted array containing numbers and a number ‘k’,
# find the first ‘k’ missing positive numbers in the array.

def firstKMissing(arr,k):
    i=0
    n=len(arr)
    while i<n:
        j=arr[i]-1
        if 0<=j<n and arr[i]!=arr[j]:
            arr[j],arr[i]=arr[i],arr[j]
        else:
            i+=1
    count=0
    extraNum=set()
    print(arr)
    for i in range(n):
        if count<k and i !=arr[i]-1 :
            print(i+1,end=" ")
            extraNum.add(arr[i])
            count+=1
    i=1
    if count<k:
        while count<k:
            num=n+i
            if not num in extraNum:
                print(num,end=" ")
                count+=1
            i+=1
arr=[-2, -3, 4]
k=2
firstKMissing(arr,k)
