# Given a sorted array of numbers, find if a given number ‘key’ is present in the array.
# Though we know that the array is sorted, we don’t know if it’s sorted in ascending or descending order.
# You should assume that the array can have duplicates.
def binary_search(arr, key):
    if len(arr)==0:
        return -1
    start = 0
    end = len(arr) - 1
    isAscending = arr[start] < arr[end]
    while start <= end:
        mid = start + (end - start) // 2
        if arr[mid] == key:
            return mid
        if isAscending:
            if key < arr[mid]:
                end = mid - 1
            else:
                start = mid + 1
        else:
            if key < arr[mid]:
                start = mid + 1
            else:
                end = mid - 1
    return -1


# arr=[1, 2, 3, 4, 5, 6, 7]
# key = 8
# print(binary_search(arr,key))

# Given an array of numbers sorted in an ascending order, find the ceiling of a given number ‘key’.
# The ceiling of the ‘key’ will be the smallest element in the given array greater than or equal to the ‘key’.
def find_ceiling(arr, key):
    start = 0
    end = len(arr) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if arr[mid] == key:
            return mid
        if key < arr[mid]:
            end = mid - 1
        else:
            start = mid + 1

        return arr[start]
    return -1


# arr=[1, 3, 8, 10, 15]
# key = 2
# print(find_ceiling(arr,key))

# Given an array of lowercase letters sorted in ascending order,
# find the smallest letter in the given array greater than a given ‘key’.
def find_ceiling_letter(arr, key):
    start = 0
    end = len(arr) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if key < arr[mid]:
            end = mid - 1
        else:
            start = mid + 1
    return arr[start % len(arr)]


# arr=['a', 'c', 'f', 'h']
# key = 'b'
# print(find_ceiling_letter(arr,key))

# Given an array of numbers sorted in ascending order, find the range of a given number ‘key’.
# The range of the ‘key’ will be the first and last position of the ‘key’ in the array.
# Write a function to return the range of the ‘key’. If the ‘key’ is not present return [-1, -1].
def find_range(arr, key, find_last_occur):
    start = 0
    end = len(arr) - 1
    ans = -1
    while start <= end:
        mid = start + (end - start) // 2
        if key == arr[mid]:
            ans = mid
            if find_last_occur:
                start = mid + 1
            else:
                end = mid - 1
        elif key < arr[mid]:
            end = mid - 1
        else:
            start = mid + 1
    return ans


def find_range_init(arr, key):
    range = [-1, -1]
    start = find_range(arr, key, False)
    if start == -1:
        return range
    else:
        range[0] = start
    range[1] = find_range(arr, key, True)
    return range

# arr = [2, 6, 6, 6, 7, 8, 9]
# key = 6
# print(find_range_init(arr, key))

def find_num(arr, key, start, end):
    while start<=end:
        mid=start +(end-start)//2
        if key==arr[mid]:
            return mid
        elif key<arr[mid]:
            end=mid-1
        else:
            start=mid+1
    return -1

def search_in_inf_arr(arr,key,n):
    start=0
    end=1
    size=2
    while end<=n and key>arr[end]:
        size*=2
        start=end+1
        end=start+size-1
    return find_num(arr,key,start,end)

# Find the maximum value in a given Bitonic array. An array is considered bitonic
# if it is monotonically increasing and then monotonically decreasing.
# Monotonically increasing or decreasing means that for any index i in the array arr[i] != arr[i+1].
def find_max_in_bitonic(arr):
    start=0
    end=len(arr)-1
    while start<end:
        mid=start + (end-start)//2
        if arr[mid]<arr[mid+1]:
            start=mid+1
        elif arr[mid]>arr[mid+1]:
            end=mid
        else:
            return mid
    return start
# arr=[1, 3, 8, 12, 4, 2]
# print(find_max_in_bitonic(arr))

#Given a Bitonic array, find if a given ‘key’ is present in it. An array is considered bitonic if
# it is monotonically increasing and then monotonically decreasing.
# Monotonically increasing or decreasing means that for any index i in the array arr[i] != arr[i+1].
def find_key_in_bitonic(arr,key):
    max_in_arr=find_max_in_bitonic(arr)
    in_asc=binary_search(arr[0:max_in_arr],key)
    if in_asc==-1:
        return max_in_arr+binary_search(arr[max_in_arr:len(arr)],key)
    return in_asc

# arr=[10, 9, 8]
# key=10
# print(find_key_in_bitonic(arr,key))

# Given an array of numbers which is sorted in ascending order and also rotated by some arbitrary number,
# find if a given ‘key’ is present in it.
def find_key_in_rotated_arr(arr,key):
    start=0
    end=len(arr)-1
    while start<=end:
        mid=(start+end)//2
        if arr[mid]==key:
            return mid
        if arr[start]<arr[mid]:
            if arr[start]<=key<arr[mid]:
                end=mid-1
            else:
                start=mid+1
        else:
            if arr[mid]<key<=arr[end]:
                start=mid+1
            else:
                end=mid-1
    return -1

# arr=[4, 5, 7, 9, 10, -1, 2]
# key = 10
# print(find_key_in_rotated_arr(arr,key))

def find_key_in_rotated_arr_having_duplicates(arr,key):
    start=0
    end=len(arr)-1
    while start<=end:
        mid=(start+end)//2
        if arr[mid]==key:
            return mid
        if arr[start]==arr[mid]==arr[end]:
            start+=1
            end-=1
        elif arr[start]<arr[mid]:
            if arr[start]<=key<arr[mid]:
                end=mid-1
            else:
                start=mid+1
        else:
            if arr[mid]<key<=arr[end]:
                start=mid+1
            else:
                end=mid-1
    return -1

# arr=[3, 5,7, 3, 3, 3]
# key = 5
# print(find_key_in_rotated_arr_having_duplicates(arr,key))

# Given an array of numbers which is sorted in ascending order and is rotated ‘k’ times around a pivot, find ‘k’.
def find_rotation_factor(arr):
    start=0
    end=len(arr)-1
    while start<=end:
        mid=(start+end)//2
        if mid>start and arr[mid]<arr[mid-1]:
            return mid
        elif mid<end and arr[mid+1]<arr[mid]:
            return mid+1
        if arr[start]<arr[mid]:
            start=mid+1
        else:
            end=mid-1
    return 0
# arr=[ 3, 8,10, 15, 1]
# print(find_rotation_factor(arr))

def find_rotation_factor_having_duplicate(arr):
    start=0
    end=len(arr)-1
    while start<=end:
        mid=(start+end)//2
        if mid>start and arr[mid]<arr[mid-1]:
            return mid
        elif mid<end and arr[mid+1]<arr[mid]:
            return mid+1
        if arr[start]==arr[mid]==arr[end]:
            if arr[start]>arr[start+1]:
                return start
            if arr[end]<arr[end-1]:
                return end
            start+=1
            end-=1
        elif arr[start]<arr[mid]:
            start=mid+1
        else:
            end=mid-1
    return 0
# arr=[3, 3,3,5, 7, 3]
# print(find_rotation_factor_having_duplicate(arr))







