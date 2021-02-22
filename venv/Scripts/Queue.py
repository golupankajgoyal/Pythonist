def quickSort(arr,start,end):
    if end-start<1:
        return
    count=partition(arr,start,end)
    i=start
    j=end
    while i<count and j>count:
        if arr[i]<=arr[count]:
            i+=1
        elif arr[j]>arr[count]:
            j-=1
        else:
            temp=arr[i]
            arr[i]=arr[j]
            arr[j]=temp
    quickSort(arr,start,count-1)
    quickSort(arr,count+1,end)

def partition(arr, start, end):
    count=start
    for i in arr[start+1:end+1]:
        if arr[start]>=i:
            count+=1
    temp=arr[count]
    arr[count]=arr[start]
    arr[start]=temp
    return count

arr=[5,8,3,7,9,5,2,6,4,10,5,15,5]
quickSort(arr,0,len(arr)-1)
print(arr)

