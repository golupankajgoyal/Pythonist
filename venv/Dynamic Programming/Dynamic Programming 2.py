import sys
from math import inf
from collections import deque,defaultdict
from operator import itemgetter,attrgetter

# Optimal Strategy for a Game
def max_profit(arr,start,end,dp):
    if start>=end:
        return 0
    if dp[start][end]!=-1:
        return dp[start][end]
    profit1=arr[start] + min(max_profit(arr,start+2,end,dp),max_profit(arr,start+1,end-1,dp))
    profit2=arr[end] + min(max_profit(arr,start+1,end-1,dp),max_profit(arr,start,end-2,dp))
    dp[start][end]=max(profit1,profit2)
    return dp[start][end]

# arr=[18, 20, 15, 30]
# dp=[[-1 for i in range(len(arr))]for j in range(len(arr))]
# ans=max_profit(arr,0,len(arr)-1,dp)
# print(ans)
# print(sum(arr)-ans)
# for i in range(len(dp)):
#     print(dp[i])

# Word Break Problem
# Given a string A and a dictionary of n words B, find out if A can be segmented
# into a space-separated sequence of dictionary words.
def break_word(string,start,end,dp,words):
    if start>end:
        return True
    if start==end:
        return string[start] in words
    if dp[start][end]!=-1:
        return dp[start][end]
    for i in range(start,end+1):
        temp=string[start:i+1] in words
        if temp and break_word(string,i+1,end,dp,words):
            dp[start][end]=True
            return True
    dp[start][end]=False
    return False

# string="ilikesamsung"
# words={"i", "like", "sam", "sung", "samsung", "mobile",
# "ice","cream", "icecream", "man", "go", "mango" }
# n=len(string)
# dp=[[-1 for i in range(n)]for j in range(n)]
# print(break_word(string,0,n-1,dp,words))


# Mobile Numeric Keypad Problem
# Given the mobile numeric keypad. You can only press buttons that are up, left, right or down to the current
# button. You are not allowed to press bottom row corner buttons (i.e. * and # ).

def find_permutations(i,j,n,grid,dp):
    if grid[i][j]==-1:
        return 0
    if n==0:
        return 1
    index=(3*i)+j+1 if grid[i][j]!=0 else 0
    if dp[n][index]!=-1:
        return dp[n][index]
    result=result=find_permutations(i,j,n-1,grid,dp)
    if j-1>=0 :
        result+=find_permutations(i,j-1,n-1,grid,dp)
    if j+1<len(grid[0]):
        result += find_permutations(i, j + 1, n - 1, grid,dp)
    if i-1>=0 :
        result += find_permutations(i-1, j, n - 1, grid,dp)
    if i+1<len(grid) :
        result += find_permutations(i+1, j , n - 1, grid,dp)
    dp[n][index]=result
    return result

def find_permutations_init(n):
    grid = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [-1,0,-1]]
    ans = 0
    dp=[[-1 for i in range(10)]for i in range(n+1)]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            temp = find_permutations(i, j, n - 1, grid,dp)
            ans += temp
    return ans
# n=5
# print(find_permutations_init(n))

# Find number of solutions of a linear equation of n variables
# Given a linear equation of n variables, find number of non-negative integer solutions of it.
# For example,let the given equation be “x + 2y = 5”, solutions of this equation are
# “x = 1, y = 2”, “x = 5, y = 0” and “x = 1. It may be assumed that all coefficients in given
# equation are positive integers.
def find_total_solution(coff,start,value,dp):
    if value<0:
        return 0
    if value==0 :
        return 1
    if dp[start][value]!=-1:
        return dp[start][value]
    ans=0
    for i in range(start,len(coff)):
        temp=find_total_solution(coff,i,value-coff[i],dp)
        ans+=temp
    dp[start][value]=ans
    return ans
# coff=[2,2,2]
# value=6
# dp=[[-1 for i in range(value+1)]for j in range(len(coff))]
# print(find_total_solution(coff,0,value,dp))

# Count number of binary strings without consecutive 1’s
class Count:
    def __init__(self,ones=0,twos=0):
        self.ones=ones
        self.twos=twos
def string_without_ones(n,dp):
    if n==1:
        return Count(1,1)
    if dp[n]!=-1:
        return dp[n]

    temp=string_without_ones(n-1,dp)
    ones=temp.twos
    twos=temp.ones+temp.twos
    dp[n]=Count(ones,twos)
    return dp[n]

def countStrings(n):
    # code here
    dp=[-1 for i in range(n+1)]
    dp[1]=Count(1,1)
    for i in range(2,n+1):
        # print(dp[i-1])
        temp=dp[i-1]
        ones=temp.twos
        twos=temp.ones+temp.twos
        dp[i]=Count(ones,twos)
    ans=dp[n]
    return (ans.ones+ans.twos)%(10**9+7)

# n=4
# dp=[-1 for i in range(n+1)]
# ans=countStrings(n)
# print(ans)

#We have to paint n boards of length {A1, A2…An}. There are k painters available and each takes 1
# unit time to paint 1 unit of board. The problem is to find the minimum time to get
# this job done under the constraints that any painter will only paint continuous sections of boards
def min_time(arr,i,k,dp):
    if i==0 and k!=0:
        return arr[i]
    if k==1:
        return sum(arr[:i+1])
    if dp[k][i]!=-1:
        return dp[k][i]
    ans=inf
    for j in range(i-1,-1,-1):
        temp=max(min_time(arr,j,k-1,dp),sum(arr[j+1:i+1]))
        ans=min(ans,temp)
    dp[k][i]=ans
    return ans
# arr=[10, 20, 30, 40]
# k=3
# dp=[[-1 for j in range(len(arr))]for i in range(k+1)]
# print(min_time(arr,len(arr)-1,k,dp))
# for i in range(k+1):
#     print(dp[i])

# Find if a string is interleaved of two other strings
def isInterleaved(string,i,j,k,s1,s2,dp):

    if i==len(string):
        if (j==len(s1) and k==len(s2)):
            return True
        else:
            return False
    if dp[j][k]!=-1:
        return dp[i][j]
    result=False
    if j<len(s1) and k<len(s2) and string[i]==s1[j] and  string[i]==s2[k]:
        result=isInterleaved(string,i+1,j+1,k,s1,s2,dp) or isInterleaved(string,i+1,j,k+1,s1,s2,dp)
    elif j<len(s1) and string[i]==s1[j] and string[i]!=s2[j]:
        result= isInterleaved(string,i+1,j+1,k,s1,s2,dp)
    elif k<len(s2) and string[i]==s2[k] and string[i]!=s2[k]:
        result= isInterleaved(string,i+1,j,k+1,s1,s2,dp)
    dp[j][k]=result
    return result

# string= "XXY"
# string1= "YX"
# string2= "X"
# dp=[[-1 for i in range(len(string2)+1)]for j in range(len(string1)+1)]
# print(isInterleaved(string,0,0,0,string1,string2,dp))

def isInterleave(s1, s2, string):
    # Code here
    dp = [[False for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    dp[0][0]=True
    for i in range(len(s1)+1):
        for j in range(len(s2)+1):
            if i==0 and j==0:
                dp[i][j]=True
            elif i == 0 :
                dp[0][j] = s2[j-1] == string[i + j-1] and dp[0][j-1]
            elif j == 0 :
                dp[i][0] = s1[i-1] == string[i + j-1] and dp[i-1][0]
            elif s1[i-1] == string[i + j-1] and s1[j-1] == string[i + j-1]:
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif s1[i-1] == string[i + j-1] and s2[j-1] != string[i + j-1]:
                dp[i][j] = dp[i - 1][j]
            elif s2[j-1] == string[i + j-1] and s1[i-1] != string[i + j-1]:
                dp[i][j] = dp[i][j-1]
    # for i in range(len(s1)+1):
    #     print(dp[i])
    return dp[len(s1)][len(s2)]
# print(isInterleave("XXY", "XXZ", "XXZXXXY"))
# print(isInterleave("XY", "WZ", "WZXY"))
# print(isInterleave("XY", "X", "XXY"))
# print(isInterleave("YX", "X", "XXY"))
# print(isInterleave("XXY", "XXZ", "XXXXZY"))

# Wildcard Pattern Matching
def wildcard_match(pattern,i,string,j,dp):
    if i==len(pattern) :
        if j==len(string):
            return True
        else:
            return False
    if j == len(string):
        for char in pattern[i:]:
            if char!="*":
                return False
        return True
    if dp[i][j]!=-1:
        return dp[i][j]
    wildcards=["*","?"]
    result=False
    if pattern[i] not in wildcards :
        if pattern[i]==string[j] :
            result=wildcard_match(pattern,i+1,string,j+1,dp)
    else:
        if pattern[i]=="?":
            result= wildcard_match(pattern,i+1,string,j+1,dp)
        else:
            if i+1<len(pattern) and pattern[i+1]=="*":
                result= wildcard_match(pattern,i+1,string,j,dp)
            else:
                r1=wildcard_match(pattern,i+1,string,j,dp)
                r2=wildcard_match(pattern,i,string,j+1,dp)
                r3=wildcard_match(pattern,i+1,string,j+1,dp)
                result= r1 or r2 or r3
    dp[i][j]=result
    return result
# Text = "baaabab"
# Pattern = "*a*****ab"
# dp=[[-1 for j in range(len(Text))]for i in range(len(Pattern))]
# print(wildcard_match(Pattern,0,Text,0,dp))
# for i in range(len(Text)):
#     print(dp[i])
# print(wildcard_match("ba*****ab",0,Text,0,dp))
# print(wildcard_match("ba*ab",0,Text,0,dp))
# print(wildcard_match("a*ab",0,Text,0,dp))
# print(wildcard_match("a*****ab",0,Text,0,dp))
# print(wildcard_match("*a*****ab",0,Text,0,dp))
# print(wildcard_match("ba*ab****",0,Text,0,dp))
# print(wildcard_match("****",0,Text,0,dp))
# print(wildcard_match("*",0,Text,0,dp))
# print(wildcard_match("aa?ab",0,Text,0,dp))
# print(wildcard_match("b*b",0,Text,0,dp))
# print(wildcard_match("a*a",0,Text,0,dp))
# print(wildcard_match("baaabab",0,Text,0,dp))
# print(wildcard_match("?baaabab",0,Text,0,dp))
# print(wildcard_match("*baaaba*",0,Text,0,dp))

#Largest sum subarray with at-least k numbers
def largest_subarray(arr,k):
    win_start=0
    max_sum=-inf
    curr_sum=-inf
    sum_arr=[0]*len(arr)
    for i in range(len(arr)):
        curr_sum=max(curr_sum+arr[i],arr[i])
        sum_arr[i]=curr_sum
    curr_sum=0
    for win_end in range(len(arr)):
        curr_sum+=arr[win_end]
        if win_end-win_start+1==k:
            max_sum=max(max_sum,curr_sum)
            if win_end-k>=0:
                max_sum=max(max_sum,sum_arr[win_end-k]+curr_sum)
            curr_sum-=arr[win_start]
            win_start+=1
    return max_sum

# arr=[-410,-349,203,-733,46,538,782,-176,681,828,282,378,-227,802,801,-774,368,261,-32,-910]
# k=12
# arr=[-4, -2, 1, -3]
# k=2
# arr=[1,-854,452,-357,-485,987,14,23,67,45,55,-528,579,-145,-52]
# k=5
# print(largest_subarray(arr,k))

# Probability of Knight
# Given an NxN chessboard and a Knight at position (x, y). The Knight has to take exactly K steps,
# where at each step it chooses any of the 8 directions uniformly at random.
# Find the probability that the Knight remains in the chessboard after taking K steps,
# with the condition that it can’t enter the board again once it leaves it.
dx=[-2,-1,1,2,-2,-1,1,2]
dy=[-1,-2,-2,-1,1,2,2,1]

def knight_prob(n,k,x,y,dp):
    if k==0 and 0<x<=n and 0<y<=n:
        return 1
    if dp[k][x][y]!=-1:
        return dp[k][x][y]
    prob=0
    for i in range(len(dx)):
        new_x= x + dx[i]
        new_y= y + dy[i]
        if 0<new_x<=n and 0<new_y<=n:
            prob+=knight_prob(n,k-1,new_x,new_y,dp)
    dp[k][x][y]=prob/8
    return dp[k][x][y]
# x=2
# y=3
# n=4
# k=4
# dp=[[[-1 for i in range(n+1)]for j in range(n+1)]for k in range(k+1)]
# prob=knight_prob(n,k,x,y,dp)
# print(prob)

# Pascal's triangle
def pascal_triangle(n):
    dp=[[0 for i in range(n+3)]for j in range(2)]
    flag=1
    dp[0][1]=1
    for i in range(2,n+1):
        for j in range(1,i+2):
            dp[flag][j]=dp[1-flag][j-1]+dp[1-flag][j]
        flag=1-flag
    print(dp[1-flag][1:n+1])

# n=6
# pascal_triangle(n)

def waterOverflow( k, r, c):
    dp=[[0 for i in range(k+1)]for j in  range(k+1)]
    dp[1][1]=k
    for i in range(1,k):
        for j in range(1,i+1):
            if dp[i][j]>1:
                dp[i+1][j]+= (dp[i][j]-1)/2
                dp[i + 1][j + 1]+= (dp[i][j]-1)/2
                dp[i][j]=1
    if type(dp[r][c]) is int:
        return (dp[r][c])
    else:
        return round(dp[r][c],6)

# k=127
# r=7
# c=1
# print(waterOverflow(k,r,c))

# Remove minimum elements from either side such that 2*min becomes more than max
def find_min_max(arr,start,end,dp):
    if start==end:
        dp[start][end]=(arr[start],arr[end])
        return dp[start][end]
    if dp[start][end]!=(0,0):
        return dp[start][end]
    _min,_max=find_min_max(arr,start,end-1,dp)
    dp[start][end]=(min(_min,arr[end]),max(_max,arr[end]))
    return dp[start][end]

def remove_elements(arr,start,end,dp,min_max):
    if start==end:
        return 0
    if dp[start][end]!=-1:
        return dp[start][end]
    _min,_max=min_max[start][end]
    if _min*2>_max:
        return 0
    dp[start][end]=min(remove_elements(arr,start+1,end,dp,min_max),remove_elements(arr,start,end-1,dp,min_max))+1
    return dp[start][end]

# arr=[4, 5, 100, 9, 10, 11, 12, 15, 200]
# dp=[[-1 for i in range(len(arr))]for  j in range(len(arr))]
# min_max=[[(0,0) for i in range(len(arr))]for  j in range(len(arr))]
# for  i in range(len(arr)):
#     find_min_max(arr,i,len(arr)-1,min_max)
# print(min_max)
# print(remove_elements(arr,0,len(arr)-1,dp,min_max))
# for i in range(len(arr)):
#     print(dp[i])

# Number of subsequences of the form a^i b^j c^k
def find_count(string,a_count,b_count,c_count,i):
    if i<0:
        return
    find_count(string,a_count,b_count,c_count,i-1)
    if string[i]=="a":
        a_count[0]+=(a_count[0]+1)
    if string[i]=="b":
        b_count[0]+=(b_count[0]+a_count[0])
    if string[i]=="c":
        c_count[0]+=(c_count[0]+b_count[0])
    return

# string="abcabc"
# a_count=[0]
# b_count=[0]
# c_count=[0]
# find_count(string,a_count,b_count,c_count,len(string)-1)
# print(c_count[0])


# Length of the longest valid substring
def longest_valid(string):
    stack=deque()
    stack.append(-1)
    ans=0
    for i in range(len(string)):
        if string[i]=="(":
            stack.append(i)
        else:
            if string[stack[-1]]=="(":
                # print(stack[-1],i)
                stack.pop()
                ans=max(ans,i-stack[-1])
            else:
                stack.append(i)
    return ans
# string="()(())("
# print(longest_valid(string))

# Boolean Parenthesization Problem
def  boolean_paranthesis(string1,string2,start,end,dp):
    if start==end:
        if string1[start]=="T":
            return [1,0]
        else:
            return [0,1]
    if dp[start][end]!=-1:
        return dp[start][end]
    count_t=0
    count_f=0
    for i in range(start+1,end+1):
        left_side=boolean_paranthesis(string1,string2,start,i-1,dp)
        right_side=boolean_paranthesis(string1,string2,i,end,dp)
        print(left_side,right_side,start,i,end)
        if string2[i-1]=="&":
            count_t+=(left_side[0]*right_side[0])
            count_f+=(left_side[1]*right_side[1]+left_side[0]*right_side[1]
                      +left_side[1]*right_side[0])
        elif string2[i-1]=="|":
            count_f += (left_side[1] * right_side[1])
            count_t += (left_side[1] * right_side[0] + left_side[0] * right_side[1]
                        + left_side[0] * right_side[0])
        elif string2[i-1]=="^":
            count_f += (left_side[1] * right_side[1]+ left_side[0] * right_side[0])
            count_t += (left_side[1] * right_side[0] + left_side[0] * right_side[1])
        print(start, end,[count_t,count_f])
    dp[start][end]=[count_t,count_f]
    return [count_t,count_f]

# symbol   = ["F", "T"]
# operator = ["|" ]
# dp=[[-1 for i in range(len(symbol))]for j in range(len(symbol))]
# print(boolean_paranthesis(symbol,operator,0,len(symbol)-1,dp))
MOD=(10**9)+7
# Count Possible Decodings of a given Digit Sequence
def decode_digits(digits,i,dp):
    if (i == 0 and digits[i]!=0 ) or i < 0:
        return 1
    if dp[i] != -1 :
        return dp[i]
    ans = 0
    if int(digits[i]) != 0:
        ans = decode_digits(digits, i - 1, dp)
    if int(digits[i-1])!=0 and 0 < int(digits[i-1:i + 1]) < 27 :
            ans += decode_digits(digits, i - 2, dp)
    dp[i] = ans
    # print(ans)
    return ans

# digits = "1411511110191111101011871111111111111511374411510811111311124711511116468111611111116111111111117"
# dp=[-1 for i in range(len(digits))]
# print(decode_digits(digits,len(digits)-1,dp)%MOD)
# print(dp)

def decode_digits_dp(digits):
    dp = [0 for i in range(len(digits))]
    if int(digits[0]) != 0:
        dp[0] = 1
    else:
        return 0
    if len(digits) > 1:
        if int(digits[1])==0:
            if int(digits[0:2])<27:
                dp[1]=1
            else:
                return 0
        elif int(digits[0:2]) <27:
            dp[1] = 2
        else:
            dp[1]=1
    else:
        return dp[0]
    for i in range(2, len(digits)):
        if int(digits[i]) != 0:
            dp[i] = dp[i - 1]
        if int(digits[i - 1]) != 0 and 0 < int(digits[i - 1:i + 1]) < 27:
            dp[i] += dp[i - 2]
    return dp[len(digits) - 1] % MOD
# digits = "1411511110191111101011871111111111111511374411510811111311124711511116468111611111116111111111117"
# # digits="10"
# print(decode_digits_dp(digits))

def perfectSum( arr, n, sum):
    # code here
    dp = [[0 for i in range(sum + 1)] for j in range(n)]
    if arr[0] <= sum:
        dp[0][arr[0]] = 1
    for i in range(n):
        dp[i][0] = 1
    curr_sum=arr[0]
    for i in range(1, n):
        curr_sum+=arr[i]
        for j in range(1, min(curr_sum,sum) + 1):
            if j - arr[i] >= 0:
                dp[i][j] = dp[i-1][j - arr[i]]
            dp[i][j] += dp[i - 1][j]
    # for i in range(len(dp)):
    #     print(dp[i])
    return dp[n - 1][sum]%MOD

# arr=[6,10,1,4,7,1,9,5,10,5,3,5,10,1,5,4,4,3,8,10,10,7,4,1,6,7,6,6,8,3,4,4,2,7,7,1,9,6,5,9,10,9,5,1,2,2,4,5,4,3,4,5,9,10,8,4,6,3,1,5,8,7,8,9,3,5,1,1,2,5,1,3,6,8,3,7,9,8,3,4,2,8,8,10,7,7,5,4,2,8,9,9,4,6,7,8]
# sum=185
# print(perfectSum( arr, len(arr), sum))

# Vertex Cover Problem
# https://www.geeksforgeeks.org/vertex-cover-problem-set-2-dynamic-programming-solution-tree/
class Node:
    def __init__(self,value):
        self.value=value
        self.dp=0
        self.left=None
        self.right=None

def find_vertex_cover(root):
    if root==None or root.left==root.right==None:
        return 0
    if  root.dp!=0:
        return root.dp
    ans1=find_vertex_cover(root.left)+find_vertex_cover(root.right)+1
    ans2=0
    if root.left:
        ans2+=(find_vertex_cover(root.left.left)+find_vertex_cover(root.left.right)+1)
    if root.right:
        ans2+=(find_vertex_cover(root.right.right)+find_vertex_cover(root.right.left)+1)
    root.dp=min(ans1,ans2)
    return root.dp

# root = Node(20)
# root.left = Node(8)
# root.left.left = Node(4)
# root.left.right = Node(12)
# root.left.right.left = Node(10)
# root.left.right.right = Node(14)
# root.right = Node(22)
# root.right.right = Node(25)
# print(find_vertex_cover(root))
# print(root.value,root.dp)

# Longest Even Length Substring such that Sum of First and Second Half is same
# In O(N^3)
def find_same_sum_string(string,start,end):
    print(start,end)
    if end==start:
        return 0
    if end-start==1:
        if string[start]==string[end]:
            return 2
        else:
            return 0
    mid=(end+start+1)//2
    if sum(string[start:mid])==sum(string[mid:end+1]):
        return end-start+1
    ans=max(find_same_sum_string(string,start+2,end),
            find_same_sum_string(string,start,end-2),
            find_same_sum_string(string,start+1,end-1))
    return ans
# str = [1,5,3,8,0,2]
# print(find_same_sum_string(str,0,len(str)-1))

#In O(N^2)
def find_string(string):
    n=len(string)
    max_len = 0
    for i in range(1, n):
        start = i - 1
        end = i
        left_sum = right_sum = 0
        while start >= 0 and end < n:
            left_sum += int(string[start])
            right_sum += int(string[end])
            start -= 1
            end += 1
            if left_sum == right_sum:
                max_len = max(max_len, end - start - 1)
    return max_len
# string="0000000"
# print(find_string(string))

# Longest Repeating Subsequence
def longest_repeating_sunsequence(i,j,string,dp):
    if i==len(string) or j==len(string):
        return 0
    if dp[i][j]!=-1:
        return dp[i][j]

    # print(i,j)
    if string[i]==string[j] and i!=j:
        result= longest_repeating_sunsequence(i+1,j+1,string,dp)+1
    else:
        result=max(longest_repeating_sunsequence(i+1,j,string,dp),
               longest_repeating_sunsequence(i,j+1,string,dp))
    dp[i][j] = result
    return result

# string="aabebcde"
# dp=[[-1 for i in range(len(string))]for j in range(len(string))]
# print(longest_repeating_sunsequence(0,0,string,dp))


# Longest Common Increasing Subsequence (LCS + LIS)
def longest_incesing_subsequence(arr1,arr2):
    lcis=[0]*len(arr2)
    for num1 in arr1:
        count=0
        for j in range(len(arr2)):
            if num1==arr2[j] and count+1>lcis[j]:
                lcis[j]=count+1
            if num1>arr2[j] and count<lcis[j]:
                count=lcis[j]
    print(lcis)
    return max(lcis)
# arr1=[3, 4, 9, 1]
# arr2=[5, 3, 8, 9, 10, 2, 1]
# print(longest_incesing_subsequence(arr1,arr2))

#Find if string is K-Palindrome or not
def k_palindromic_subseq(string,start,end,dp):
    if start==end:
        return 1
    if end-start==1 and string[start]==string[end]:
        return 2
    if dp[start][end]!=-1:
        return dp[start][end]

    if string[start]==string[end]:
        return k_palindromic_subseq(string,start+1,end-1,dp)+2

    ans=max(k_palindromic_subseq(string,start+1,end,dp),k_palindromic_subseq(string,start,end-1,dp))
    dp[start][end]=ans
    return ans
# string="abcdecba"
# dp=[[-1 for i in range(len(string))]for j in range(len(string))]
# result=k_palindromic_subseq(string,0,len(string)-1,dp)
# k=1
# if result+k>=len(string):
#     print("YES")
# else:
#     print("NO")


def k_palindromic_subseq_dp(string,n):
    dp=[[0 for i in range(n)]for j in range(n)]

    for end in range(n):
        for start in range(end,-1,-1):
            if start==end:
                dp[end][start]=1
            elif end-start==1 and string[start]==string[end]:
                print(start,end)
                dp[end][start]=2
            elif string[start]==string[end]:
                print(start,end)
                dp[end][start]=dp[end-1][start+1]+2
            else:
                dp[end][start]=max(dp[end-1][start],dp[end][start+1])
    for i in range(n):
        print(dp[i])
    return dp[n-1][0]

# string="acdcb"
# print(k_palindromic_subseq_dp(string,len(string)))

#Count Distinct Subsequences
def distinct_sequence(string,i,dp,map):
    if i==-1:
        return 1
    if i==0:
        map[string[0]] = i
        return 2
    if dp[i]!=-1:
        return dp[i]
    temp=distinct_sequence(string,i-1,dp,map)
    char = string[i]
    if map[char]==-1:
        map[char]=i
        dp[i]=2*temp
        return dp[i]

    temp=2*temp - distinct_sequence(string,map[char],dp,map)+distinct_sequence(string,map[char]-1,dp,map)
    map[char]=i
    dp[i]=temp
    return temp

# str = "bbb"
# map=defaultdict(lambda :-1)
# dp=[-1 for i in range(len(str))]
# print(distinct_sequence(str,len(str)-1,dp,map))

#Count ways to assign unique cap to every person
# There are 100 different types of caps each having a unique id from 1 to 100.
# Also, there are ‘n’ persons each having a collection of a variable number of caps.
# One day all of these persons decide to go in a party wearing a cap but to look unique they
# decided that none of them will wear the same type of cap. So, count the total number of
# arrangements or ways such that none of them is wearing the same type of cap.

# Brute-Force Solution
def assign_caps(people,i,map):
    if i==len(people):
        return 1
    ans=0
    print(i,map)
    caps=people[i]
    for cap in caps:
        if map[cap]==1:
            map[cap]=0
            ans+=(assign_caps(people,i+1,map))
            map[cap]=1

    return ans if ans!=0 else -1

# map=defaultdict(lambda :1)
# people=defaultdict((list))
# people[0]=[5, 100, 1]
# people[1]=[2]
# people[2]=[5, 100]
# print(assign_caps(people,0,map))

#Maximum sum increasing subsequence
def max_sum_incresing_subseq(arr,n):
    sum_arr=[0]*n
    sum_arr[0]=arr[0]
    for i in range(1,n):
        temp=0
        for j in range(i):
            if arr[j]<arr[i]:
                temp=max(temp,sum_arr[j])
        sum_arr[i]=temp+arr[i]
    print(sum_arr)
    return max(sum_arr)

# arr=[1, 101, 2, 3, 100, 4, 5]
# print(max_sum_incresing_subseq(arr,len(arr)))

# Arpa and choosing brother free subset
# https://www.hackerearth.com/practice/algorithms/dynamic-programming/bit-masking/practice-problems/algorithm/brofree-34df073d/

def choose_member(i,k,arr,dp):
    if i<0:
        return 0
    if k==0:
        return 1
    if k==1 and i==0:
        return max(arr[0],1)
    if dp[i][k]!=-1:
        return dp[i][k]
    ans=choose_member(i-1,k,arr,dp)
    if arr[i]!=0:
        ans+=(choose_member(i-1,k-1,arr,dp)*arr[i])
    dp[i][k]=ans
    # print(i, k,ans)
    return ans

# arr=[1 ,2 ,3 ,2 ,3]
# n=len(arr)
# k=2
# map=[0]*(n+1)
# for i in arr:
#     map[i]+=1
# print(map)
# dp=[[-1 for i in range(k+1)]for j in range(n+1)]
# print(choose_member(n-1,k,map,dp))

# Program for nth Catalan Number

def catlan_number(n,dp):
    if n==0:
        return 1
    





































