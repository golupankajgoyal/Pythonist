from math import inf,log
from collections import defaultdict,deque
from heapq import heapify,heappush as hpush,heappop as hpop,heapreplace as hreplace
import sys

# 938. Range Sum of BST
def bst_sum_in_range(root,low,high, sum):
    if root is None:
        return 0
    num=root.val
    if num>=low and num<=high :
        sum[0]+=num
    if num>low:
        bst_sum_in_range(root.left,low,min(high,num),sum)
    if num<high:
        bst_sum_in_range(root.right,max(low,num),high,sum)
    return

# 897. Increasing Order Search Tree
def find_leftmost(node):
    if node==None:
        return None
    if node.left==None:
        return node.val
    return find_leftmost(node.left)


def increasing_bst(old_node,new_node):
    if old_node==None:
        return
    increasing_bst(old_node.left,new_node)
    new_node[0].right=TreeNode(old_node.val)
    new_node[0]=new_node.right
    increasing_bst(old_node.right,new_node)

# 559. Maximum Depth of N-ary Tree
def depth_n_array(root):
    if root == None:
        return 0
    max_depth = 0 if (not root.children ) else max([depth_n_array(child) for child in root.children])
    return max_depth + 1

# 108. Convert Sorted Array to Binary Search Tree
def sortedArrToBst(start,end,arr):
    if start>end:
        return None
    mid = (start + end)//2
    root=TreeNode(arr[mid])
    root.left=sortedArrToBst(start,mid-1,arr)
    root.right=sortedArrToBst(mid+1,end,arr)
    return root

# 690. Employee Importance
def dfs(graph,id):
    ans=graph[id][1]
    for child in graph[id][0]:
        ans+=dfs(graph,child)
    return ans

def buildGraph(employees,id):
    graph=defaultdict(defaultdict)
    for employee in employees:
        info=defaultdict()
        info[0]=employee[2]
        info[1]=employee[1]
        graph[employee[0]]=info
    return dfs(graph,id)

# employees=[[1, 5, [2, 3]], [2, 3, []], [3, 3, []]]
# print(buildGraph(employees,1))

# 733. Flood Fill
def floodFill(startRow,startCol,grid,newColor,oldColor):
    if startRow<0 or startRow>=len(grid) or startCol<0 or startCol>=len(grid[0])\
            or grid[startRow][startCol]!=oldColor:
        return

    currColor=grid[startRow][startCol]
    grid[startRow][startCol]=newColor
    points = [(-1,0),(0,-1),(0,1),(1,0)]
    for r,c in points:
        floodFill(startRow+r,startCol+c,grid,newColor,currColor)

# image = [[1,1,1],[0,1,0],[1,0,1]]
# sr = 1
# sc = 1
# newColor = 2
# print(image)
# floodFill(sr,sc,image,newColor,image[sr][sc])
# print(image)

# 110. Balanced Binary Tree
def balanceBinaryTree(root,result):
    if root == None:
        return 0
    leftHeight=balanceBinaryTree(root.left,result)
    rightHeight=balanceBinaryTree(root.right, result)
    if result[0] == True and abs(leftHeight-rightHeight)>1:
        result[0] = False
    return max(leftHeight,rightHeight)+1

# 112. Path Sum
def findPath(root,sum):
    if root == None:
        if sum == 0:
            return True
        else:
            return False
    return findPath(root.left,sum - root.val) or findPath(root.right, sum - root.val)

# 111. Minimum Depth of Binary Tree
def minDepth(root):
    if root == None:
        return inf
    if root.left == None and root.right == None:
        return 1
    return 1 + min(minDepth(root.left),minDepth(root.right))

# 797. All Paths From Source to Target
def allPathsSourceTarget(node,graph,currPath,result):
    if node == len(graph)-1:
        result.append(currPath)
        return

    for child in graph[node]:
        newPath=list(currPath)
        newPath.append(child)
        allPathsSourceTarget(child,graph,newPath,result)

# graph=[[4,3,1],[3,2,4],[3],[4],[]]
# result=[]
# allPathsSourceTarget(0,graph,[0],result)
# print(result)

# 1448. Count Good Nodes in Binary Tree
def goodNodes( node, parent, dp):
    if node == None:
        return

    dp[node]=max(dp[parent],node.val)
    goodNodes(node.left, node, dp)
    goodNodes(node.right, node, dp)
    return

# dp = defaultdict(lambda :-inf)
# goodNodes(root,root,dp)
# count = 0
# for key in dp:
#     if key.val >= dp[key]:
#         count += 1
# print(count)





















