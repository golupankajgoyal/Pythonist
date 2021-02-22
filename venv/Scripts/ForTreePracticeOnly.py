from collections import deque
import sys
class Node:
    def __init__(self, data):
        self.data = data
        self.left =None
        self.right = None

def inOrderPrint(root):
    if root is None:
        return
    inOrderPrint(root.left)
    print(root.data,end=" ")
    inOrderPrint(root.right)

def levelOrder(root):
    q=deque()
    q.append(root)
    while q:
        l=len(q)
        for _ in range(l):
            curr=q.popleft()
            print(curr.data,end=" ")
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
        print()

def appendNode(root,node):
    if root is None:
        return node
    if root.data>node.data:
        root.left=appendNode(root.left,node)
    else:
        root.right=appendNode(root.right,node)
    return root

def makeBstFromUnsortedArray(arr):
    if len(arr)<1:
        return None
    root=Node(len(arr[0]))
    for i in arr[1:]:
        node=Node(len(i))
        root=appendNode(root,node)
    return root


root = Node(100)
root.left = Node(-2)
root.right = Node(125)

root.left.left = Node(-8)
root.left.right = Node(75)
root.right.left = Node(112)
root.right.right = Node(150)

root.left.left.left = Node(-12)
root.left.left.right = Node(-5)
# ///////////////////////////////////////
root2 = Node(80)

root2.left = Node(40)
root2.right = Node(95)

root2.left.left = Node(20)
root2.left.right = Node(55)
root2.right.left = Node(85)
root2.right.right = Node(101)

root2.left.left.left = Node(15)
root2.left.left.right = Node(25)

# Given a binary tree, populate an array to represent its level-by-level traversal.
# You should populate the values of all nodes of each level from left to right in separate sub-arrays.
def bfs_traversal(root):
    output=[]
    if root is None:
        return
    queue=deque()
    queue.append(root)
    while queue:
        level_size=len(queue)
        curr_level=[]
        for i in range(level_size):
            curr=queue.popleft()
            curr_level.append(curr.data)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        output.append(curr_level)
    print(output)

# bfs_traversal(root2)

# Given a binary tree and a number ‘S’,
# find all paths from root-to-leaf such that the sum of all the node values of each path equals ‘S’.
def find_all_paths(node,sum,curr_path,all_path):
    if node is None:
        return False
    curr_path.append(node.data)
    if node.data== sum and node.left==None and node.right==None:
        all_path.append(list(curr_path))
    else:
        find_all_paths(node.left,sum-node.data,curr_path,all_path)
        find_all_paths(node.right,sum-node.data,curr_path,all_path)

    del curr_path[-1]

def find_all_path(root,sum):
    all_paths=[]
    find_all_paths(root,sum,[],all_paths)
    return all_paths

# print(find_all_path(root2,155))

def find_max_paths(node,curr_sum,curr_path,max_sum,max_path):

    if node is None:
        return False
    curr_path.append(node.data)
    curr_sum=curr_sum+node.data
    if node.left==None and node.right==None and max_sum[0]<curr_sum:
        max_path[0]=list(curr_path)
        max_sum[0]=curr_sum
    else:
        find_max_paths(node.left,curr_sum,curr_path,max_sum,max_path)
        find_max_paths(node.right,curr_sum,curr_path,max_sum,max_path)
    del curr_path[-1]

def find_max_path(root):
    max_sum=[0]
    max_path=[[]]
    find_max_paths(root,0,[],max_sum,max_path)
    return max_path[0],max_sum[0]

# print(find_max_path(root2))

# Given a binary tree where each node can only have a digit (0-9) value,
# each root-to-leaf path will represent a number. Find the total sum of all the numbers represented by all paths.
def total_sum(node,curr_num,total):
    if node is None:
        return
    curr_num=(curr_num*10)+node.data
    if node.left is None and node.right is None:
        total[0]+=curr_num
    else:
        total_sum(node.left,curr_num,total)
        total_sum(node.right,curr_num,total)


def findSubsequence(node,arr):
    if node is None:
        return len(arr)==0
    if len(arr)<=0:
        return False

    if node.data==arr[0]:
        return findSubsequence(node.left,arr[1:]) or findSubsequence(node.right,arr[1:])
    else:
        return False

# root1 = Node(1)
#
# root1.left = Node(0)
# root1.right = Node(1)
#
# root1.left.left = Node(1)
# root1.right.left = Node(6)
# root1.right.right = Node(5)
# print(findSubsequence(root1,[1,0]))

# Given a binary tree and a number ‘S’, find all paths in the tree
# such that the sum of all the node values of each path equals ‘S’.
# Please note that the paths can start or end at any node but all paths must follow
# direction from parent to child (top to bottom).
def findAllPaths(node,sum,curr_path):
    if node is None:
        return 0
    curr_path.append(node.data)
    path_sum=0
    path_count=0
    for value in curr_path[::-1]:
        path_sum+=value
        if path_sum==sum:
            path_count+=1
    path_count+=(findAllPaths(node.left,sum,curr_path)+findAllPaths(node.right,sum,curr_path))
    del curr_path[-1]
    return path_count

def findAllPathsInit(root,sum):
    return findAllPaths(root,sum,[])

# root1 = Node(12)
#
# root1.left = Node(7)
# root1.right = Node(1)
#
# root1.left.left = Node(4)
# root1.right.left = Node(10)
# root1.right.right = Node(5)
# print(findAllPathsInit(root1,11))

# Given a binary tree, find the length of its diameter.
# The diameter of a tree is the number of nodes on the longest path between any two leaf nodes.
# The diameter of a tree may or may not pass through the root.
def find_diameter(node,dia_size):
    if node is None:
        return 0
    left_len=find_diameter(node.left,dia_size)
    right_len=find_diameter(node.right,dia_size)
    curr_diameter=1+ left_len+ right_len
    if curr_diameter>dia_size[0]:
        dia_size[0]=curr_diameter
    return 1+max(left_len,right_len)

def find_diameter_init(root):
    dia_size=[0]
    find_diameter(root,dia_size)
    return dia_size[0]

# print(find_diameter_init(root2))


def findMaxSum(node,max_sum):
    if node is None:
        return 0
    left_sum=findMaxSum(node.left,max_sum)
    right_sum=findMaxSum(node.right,max_sum)
    curr_sum=left_sum+right_sum+node.data
    max_sum[0]=max(max_sum[0],curr_sum)
    return max(left_sum,right_sum)+node.data

def findMaxSumInit(root):
    max_sum=[0]
    findMaxSum(root,max_sum)
    return max_sum[0]
print(findMaxSumInit(root2))





