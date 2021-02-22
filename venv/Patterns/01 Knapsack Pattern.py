
# Given a set of positive numbers, find if we can partition it into two subsets
# such that the sum of elements in both subsets is equal.
def find_set_of_equal_sum(arr,sum,i):
    if sum==0:
        return True
    if i == len(arr):
        return False
    return find_set_of_equal_sum(arr,sum-arr[i],i+1) or find_set_of_equal_sum(arr,sum,i+1)

def find_set_of_equal_sum_helper(arr):
    s=sum(arr)
    if s%2!=0:
        return False
    else:
        return find_set_of_equal_sum(arr,s//2,0)


def find_set_of_equal_sum_mem(arr,dp,s,i):
    if i==len(arr) or s<0:
        if s==0:
            return True
        else:
            return False
    if dp[i][s]!=-1:
        return dp[i][s]
    result=find_set_of_equal_sum_mem(arr,dp,s,i+1) or find_set_of_equal_sum_mem(arr,dp,s-arr[i],i+1)
    dp[i][s]=result
    return result

def find_set_of_equal_sum_mem_helper(arr):
    s=sum(arr)
    dp=[[-1 for j in range((s//2)+1)]for i in range(len(arr))]
    if s%2!=0:
        return False
    else:
        result= find_set_of_equal_sum_mem(arr,dp,s//2,0)
        for i in range(len(dp)):
            print(dp[i])
        return result
# arr=[1, 1, 3, 4, 7]
# print(find_set_of_equal_sum_mem_helper(arr))

# Given a set of positive numbers, partition the set into two subsets with
# minimum difference between their subset sums.

def find_subset_of_min_diff(arr,dp,sum,i):
    if i == len(arr):
        return 0
    if dp[i]!=-1:
        return dp[i]

    max_sum=find_subset_of_min_diff(arr,dp,sum,i+1)
    if max_sum+arr[i]<=sum:
        dp[i]=max_sum+arr[i]
    else:
        dp[i]=max_sum
    return dp[i]

# arr=[1, 2, 7, 1, 5]
# s=sum(arr)
# dp=[-1 for i in range(s//2+1)]
# temp=find_subset_of_min_diff(arr,dp,s//2,0)
# print(s-2*temp)
# print(dp,s//2)

# Given a set of positive numbers, find the total number of subsets whose sum is equal to a given number ‘S’.
def find_all_subset_of_sum(arr,dp,sum,i):
    if i==len(arr) or sum<0:
        if sum==0:
            return 1
        else:
            return 0
    # print("Hii")
    if dp[i][sum]!=-1:
        return dp[i][sum]

    s1=find_all_subset_of_sum(arr,dp,sum,i+1)
    s2=find_all_subset_of_sum(arr,dp,sum-arr[i],i+1)
    dp[i][sum]=s1+s2
    return dp[i][sum]

# arr=[1, 2, 7, 1, 5]
# sum=9
# dp=[[-1 for j in range(sum+1)]for i in range(len(arr))]
# print(find_all_subset_of_sum(arr,dp,sum,0))
# for i in range(len(dp)):
#     print(dp[i])

# You are given a set of positive numbers and a target sum ‘S’.
# Each number should be assigned either a ‘+’ or ‘-’ sign.
# We need to find the total ways to assign symbols to make the sum of the numbers equal to the target ‘S’.

def find_all_possible_subset(arr,sum,num,i):
    # Read the article last problem of 0-1 knapsack pattern
    if i == len(arr):
        if sum==num:
            return 1
        else:
            return 0
    s1=find_all_possible_subset(arr,sum-arr[i],num,i+1)
    s2=find_all_possible_subset(arr,sum+arr[i],num,i+1)
    return s1+s2
arr=[1, 2, 7, 1]
num=9
print(find_all_possible_subset(arr,0,num,0))





