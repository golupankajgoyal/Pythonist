from math import inf

# 0/1 Knapsack
def knapsack(weight, profit, capacity):

    n= len(weight)
    dp = [[0]*n for i in range(capacity + 1)]
    def knapsackRecursive(pos, curr_capacity):
        if pos >= len(weight):
            return 0

        if dp[curr_capacity][pos] != 0:
            return dp[curr_capacity][pos]

        max_profit = knapsackRecursive(pos+1, curr_capacity)
        if curr_capacity - weight[pos] >= 0:
            max_profit = max(max_profit,profit[pos] + knapsackRecursive(pos+1, curr_capacity - weight[pos]))
        dp[curr_capacity][pos] = max_profit
        return max_profit
    ans = knapsackRecursive(0,capacity)
    print(dp)
    return ans

# weight = [1, 2, 3, 5]
# profit = [1, 6, 10, 16]
# capacity = 7
# print(knapsack(weight, profit, capacity))

# Equal Subset Sum Partition
def equalSumSubsets(arr):
    n= len(arr)
    target = sum(arr)
    if target%2 != 0:
        return False
    target//=2
    dp = [[0]*(target+1) for i in range(n)]

    def equalSumSubsetsDp(pos, required_sum):
        if pos == n:
            return required_sum == 0
        if dp[pos][required_sum] != 0:
            return dp[pos][required_sum]
        dp[pos][required_sum] = equalSumSubsetsDp(pos+1,required_sum) or equalSumSubsetsDp(pos+1, required_sum - arr[pos])
        return dp[pos][required_sum]
    return equalSumSubsetsDp(0,target)


# print(equalSumSubsets([1, 1, 3, 4, 7]))

# Minimum Subset Sum Difference
def minimumSubsetSumDiffrence(arr):
    n = len(arr)
    total = sum(arr)
    target = total//2

    def mSSD(pos, curr_sum):
        if pos == n:
            if curr_sum<= target:
                return curr_sum
            else:
                return -inf
        return max(mSSD(pos+1,curr_sum), mSSD(pos+1,curr_sum + arr[pos]))

    sum1 = mSSD(0,0)
    return total - 2*sum1

# print(minimumSubsetSumDiffrence([1, 2, 3, 9]))

# Count of Subset Sum
def countSubsetOfSum(arr, num):
    n = len(arr)
    dp = [[-1]*(num+1) for i in range(n)]
    def countSubsetOfSumDp(index, required_sum):
        if index == n-1 :
            ans = arr[index] == required_sum or required_sum == 0
            return ans
        if dp[index][required_sum] != -1:
            return dp[index][required_sum]

        result = countSubsetOfSumDp(index+1, required_sum - arr[index]) + countSubsetOfSumDp(index+1, required_sum)
        dp[index][required_sum] = result
        return result
    return countSubsetOfSumDp(0, num)

# print(countSubsetOfSum([1, 2, 7, 1, 5], 9))


# Unbounded Knapsack
def unboundKnapsack(profit, weight, capacity):
    n = len(profit)
    dp = [[0]*(capacity+1) for i in range(n)]

    def unboundKnapsackDp(index, curr_capacity):
        if index >= n:
            return 0
        if dp[index][curr_capacity] != 0:
            return dp[index][curr_capacity]

        ans = 0
        if weight[index] <= curr_capacity:
            ans = profit[index] + unboundKnapsackDp(index, curr_capacity - weight[index])

        ans = max(ans, unboundKnapsackDp(index+1, curr_capacity))
        dp[index][curr_capacity] = ans
        return ans
    return unboundKnapsackDp(0, capacity)

# print(unboundKnapsack([15, 20, 50], [1, 2, 3 ], 5))





















