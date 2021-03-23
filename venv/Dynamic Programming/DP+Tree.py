from math import inf,log
from collections import defaultdict,deque
from heapq import heapify,heappush as hpush,heappop as hpop,heapreplace as hreplace
class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

# Maximum Path Sum in a Binary Tree
# Given a binary tree, find the maximum path sum. The path may start and end at any node in the tree.
def max_sum(node,result):
    if node==None:
        return 0
    left_sum=max_sum(node.left,result)+node.value
    right_sum=max_sum(node.right,result)+node.value
    print(node.value,left_sum,right_sum,result)
    ans=max(left_sum,right_sum)
    result[0]=max(result[0],left_sum+right_sum-node.value)
    return max(0,ans)

# root = Node(10)
# root.left = Node(2)
# root.right   = Node(-10)
# root.left.left  = Node(20)
# root.left.right = Node(1)
# root.right.right = Node(-25)
# root.right.right.left   = Node(3)
# root.right.right.right  = Node(4)
# result=[-inf]
# print(max_sum(root,result))
# print(result[0])

# Suboordinates
# Given the structure of a company, your task is to calculate for each employee the number of their subordinates.
def count_subordinates(node,map,tree):
    if node==None:
        return
    ans=0
    for child in tree[node]:
        count_subordinates(child,map,tree)
        ans+=(map[child]+1)
    map[node]=ans

# arr=[1, 1, 2, 3]
# n=len(arr)
# tree=defaultdict(list)
# for i in range(n):
#     tree[arr[i]].append(i+2)
# map=[0]*(n+2)
# count_subordinates(1,map,tree)
# print(map[1:])

# Tree Matching
# Question : https://cses.fi/problemset/task/1130/
# Resource : https://www.youtube.com/watch?v=RuNAYVTn9qM&list=PLb3g_Z8nEv1j_BC-fmZWHFe6jmU_zv-8s&index=2
def tree_matching(node,dp,tree):
    if tree[node]==[]:
        return [0,0]
    if dp[node]!=-1:
        return dp[node]
    arr=[0,0]
    for child in tree[node]:
        arr[0]+=max(tree_matching(child,dp,tree))

    for child in tree[node]:
        temp=tree_matching(child,dp,tree)
        # print(child,temp)
        if temp[1]>temp[0]:
            arr[1]=max(arr[1],(arr[0]-temp[1]+temp[0]))
        else:
            arr[1] = max(arr[1], arr[0])
    arr[1]+=1
    dp[node]=arr
    return arr

# n=int(input())
# tree=defaultdict(list)
# for _ in range(n-1):
#     n1,n2=map(int,input().split())
#     tree[n1].append(n2)
# dp=[-1 for i in range(n+1)]
# print(tree_matching(1,dp,tree))

# input
# 10
# 6 3
# 2 8
# 10 8
# 4 10
# 7 3
# 8 7
# 1 5
# 5 9
# 10 9

# Diameter
# https://cses.fi/problemset/task/1131
def diameter(node,tree,ans,parent):
    if node==None:
        return 0
    min_heap=[0,0]
    for child in tree[node]:
        if child!=parent:
            child_len=diameter(child,tree,ans,node)
            hreplace(min_heap,child_len) if child_len>min_heap[0] else None

    if sum(min_heap)+1>ans[0]:
        ans[0]=sum(min_heap)+1
    return min_heap[1]+1

# n=int(input())
# tree=defaultdict(list)
# for _ in range(n-1):
#     n1,n2=map(int,input().split())
#     tree[n1].append(n2)
#     tree[n2].append(n1)
# ans=[0]
# diameter(1,tree,ans,0)
# print(ans[0]-1)

# Tree Distances I
# You are given a tree consisting of n nodes.
# Your task is to determine for each node the maximum distance to another node.

# def calc_depth(node,tree,depth,parent):
#     if len(tree[node])==1 and tree[node][0]==parent:
#         return 0
#     if depth[node]!=0:
#         return depth[node]
#     curr_depth=0
#     for child in tree[node]:
#         if child!=parent:
#             curr_depth=max(curr_depth,calc_depth(child,tree,depth,node))
#     curr_depth+=1
#     depth[node]=curr_depth
#     return depth[node]
#
# def calc_dist(node,tree,depth,dist,parent):
#
#     if dist[node]!=-1:
#         return dist[node]
#     for child in tree[node]:
#         if child!=parent:
#             calc_dist(child,tree,depth,dist,node)
#     ans=0
#     for child in tree[parent]:
#         if child!=node:
#             ans=max(ans,depth[child])
#     print(node,ans)
#     if depth[parent]==ans+1:
#         ans=2+ans
#     else:
#         ans=depth[node]
#     dist[node]=ans
#     return
#
# n=int(input())
# tree=defaultdict(list)
# for _ in range(n-1):
#     n1,n2=map(int,input().split())
#     tree[n1].append(n2)
#     tree[n2].append(n1)
# depth=[0 for i in range(n+1)]
# calc_depth(1,tree,depth,0)
# print(depth)
# dist=[-1 for i in range(n+1)]
# calc_dist(1,tree,depth,dist,0)
# print(dist)

#You are given a tree consisting of n nodes.
# Your task is to determine for each node the sum of the distances from the node to all other nodes.

# This funtion will calculate the distance of every subtree and total number of nodes in a particular subtree
def calc_dist_sum(node,tree,dist,nodes,parent):
    if len(tree[node])==1 and tree[node][0]==parent:
        dist[node]=0
        nodes[node]=1
        return
    if dist[node]!=0:
        return dist[node]
    curr_dist=0
    total_nodes=0
    for child in tree[node]:
        if child != parent:
            calc_dist_sum(child,tree,dist,nodes,node)
            curr_dist+=(dist[child]+nodes[child])
            total_nodes+=nodes[child]
    dist[node]=curr_dist
    nodes[node]=total_nodes+1
    return

def calc_dist_sum_from_every_node(node,tree,ans,dist,nodes,parent):

    curr_dist=dist[node]
    partial_ans = ans[parent] - dist[node] - nodes[node]
    curr_dist += (partial_ans + nodes[1] - nodes[node])
    ans[node]=curr_dist
    for child in tree[node]:
        if child!=parent:
            calc_dist_sum_from_every_node(child, tree, ans, dist, nodes, node)

# n=int(input())
# tree=defaultdict(list)
# for _ in range(n-1):
#     n1,n2=map(int,input().split())
#     tree[n1].append(n2)
#     tree[n2].append(n1)
# dist=[0 for i in range(n+1)]
# nodes=[0 for i in range(n+1)]
# calc_dist_sum(1,tree,dist,nodes,0)
# ans=[0 for  i in range(n+1)]
# ans[0]=dist[1]+nodes[1]
# calc_dist_sum_from_every_node(1,tree,ans,dist,nodes,0)
# for i in range(1,len(ans)):
#     print(ans[i],end=" ")

# Nodes in a Tree at a distance k
# https://codeforces.com/contest/161/problem/D

# This funtion will calculate the no. of nodes at distance upto k
def calc_dist_upto_k(node,parent,tree,dist,k):
    if len(tree[node])==1 and tree[node][0]==parent:
        dist[node]=[0]*(k+1)
        dist[node][0]=1
        return
    ans=[0]*(k+1)
    ans[0]=1
    for child in tree[node]:
        if child !=parent:
            calc_dist_upto_k(child,node,tree,dist,k)
            for i in range(1,k+1):
                ans[i]+=dist[child][i-1]
    dist[node]=ans
    return

def calc_nodes_pair(node,parent,tree,dist,ans):
    children=tree[node]
    count = dist[node][k]
    for i in range(len(children)):
        if children[i]!=parent:
            temp=0
            for j in range(i+1,len(children)):
                for num  in range(k-1):
                    temp+=(dist[children[i]][num]*dist[children[j]][k-num-2])
                    # print(i,j,temp)
            count+=temp
            calc_nodes_pair(children[i], node, tree, dist, ans)
    ans[node]=count
    return count

# n,k=map(int,input().split())
# tree=defaultdict(list)
# for _ in range(n-1):
#     n1,n2=map(int,input().split())
#     tree[n1].append(n2)
#     tree[n2].append(n1)
# dist=defaultdict(list)
# calc_dist_upto_k(1,0,tree,dist,k)
# ans=[0 for i in range(n+1)]
# calc_nodes_pair(1,0,tree,dist,ans)
# print(sum(ans))

# Company Queries I
# Using Binary Lifting Technique
# print(log(10**6,2))
def binary_lift(node,parent,tree,dp):
    dp[node]=[-1]*20
    dp[node][0]=parent
    for i in range(1,20):
        if dp[dp[node][i-1]]!=[]:
            dp[node][i]=dp[dp[node][i-1]][i-1]
        else:
            break
    for child in tree[node]:
        if child !=parent:
            binary_lift(child,node,tree,dp)

def bit_set_stack(pos,num,stack):
    if num==0:
        return
    if (num&1)!=0:
        stack.append(pos)
    bit_set_stack(pos+1,num>>1,stack)

def find_kth_parent(node,dp,bit_stack):
    temp=node
    while stack:
        power=stack.pop()
        temp=dp[temp][power]
    return temp

# dp=defaultdict(list)
# tree=defaultdict(list)
# n,q=map(int,input().split())
# dp[1]=[-1]*(20)
# # dp[1][0]=1
# parents=list(map(int,input().split()))
# for i in range(len(parents)):
#     tree[parents[i]].append(i+2)
#     tree[i+2].append(parents[i])
# binary_lift(1,-1,tree,dp)
# # for  i in range(1,len(dp)):
# #     print(dp[i])
# for i in range(q):
#     node,k=map(int,input().split())
#     stack = deque()
#     bit_set_stack(0, k, stack)
#     print(find_kth_parent(node,dp,stack))

# Finding LCA in O((log(n))^2)
def find_levels(node,parent,tree,levels,i):
    levels[node]=i
    for child in tree[node]:
        if child!=parent:
            find_levels(child,node,tree,levels,i+1)

def find_parent(node,parent,tree,dp):
    dp[node] = [-1] * 20
    dp[node][0]=parent
    for i in range(1,20):
        if dp[dp[node][i-1]]!=[]:
            dp[node][i]=dp[dp[node][i-1]][i-1]
        else:
            break
    for child in tree[node]:
        if child!=parent:
            find_parent(child,node,tree,dp)

def lift(node,k):
    for  i in range(20,-1,-1):
        if node==-1 or k==0:
            break
        if (k&(1<<i))!=0:
            k=k&(~(1<<i))
            node=dp[node][i]
    return node

def find_lca(u,v,tree,level):
    if level[u]<level[v]:
        u,v=v,u
    u=lift(u,level[u]-level[v])
    start,end=0,level[u]
    while start!=end:
        mid=(end+start)//2
        x1=lift(u,mid)
        x2=lift(v,mid)
        if x1==x2:
            end=mid
        else:
            start=mid+1
    return lift(u,start)

# dp=defaultdict(list)
# tree=defaultdict(list)
# n,q=map(int,input().split())
# dp[1]=[-1]*(20)
# levels=[-1 for i in range(n+1)]
# parents=list(map(int,input().split()))
# for i in range(len(parents)):
#     tree[parents[i]].append(i+2)
#     tree[i+2].append(parents[i])
# find_parent(1,-1,tree,dp)
# find_levels(1,-1,tree,levels,0)
# result=lift(4,0)
# for  _ in range(q):
#     u,v=map(int,input().split())
#     print(find_lca(u,v,tree,levels))

# Binary Tree Cameras
# https://leetcode.com/problems/binary-tree-cameras/
def find_min_cameras(node,parent_cam,node_cam,dp):
    if  node==None:
        return inf
    if node.left==None and node.right==None:
        dp[node][1][1]=1
        dp[node][1][0]=0
        dp[node][0][1]=1
        dp[node][0][0]=inf
        return dp[node][parent_cam][node_cam]

    if dp[node][parent_cam][node_cam]!=-1:
        return dp[node][parent_cam][node_cam]

    if node_cam==1:
        result=min(find_min_cameras(node.left,1,1,dp),find_min_cameras(node.left,1,0,dp))\
               +min(find_min_cameras(node.right,1,1,dp),find_min_cameras(node.right,1,0,dp))+1
    else:
        if parent_cam==1:
            result = min(find_min_cameras(node.left, 0, 1, dp), find_min_cameras(node.left, 0, 0, dp)) \
                     + min(find_min_cameras(node.right,0, 1, dp), find_min_cameras(node.right,0, 0, dp))
        else:
            result = min(find_min_cameras(node.right,0,1,dp)+min(find_min_cameras(node.left, 0, 1, dp),
                                                                 find_min_cameras(node.left, 0, 0, dp)),
                         find_min_cameras(node.left,0,1,dp) + min(find_min_cameras(node.right, 0, 1, dp),
                                                                find_min_cameras(node.right, 0, 0, dp)))
    dp[node][parent_cam][node_cam]=result
    return result

# dp=defaultdict(lambda :[[-1 for i in range(2)]for j in range(2)])
# result=min(find_min_cameras(root,0,0,dp),find_min_cameras(root,0,1,dp))



















































