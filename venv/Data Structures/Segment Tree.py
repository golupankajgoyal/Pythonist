import sys
# Segment Tree for storing the sum of elements

def update(tree,l,r,treeNode,value,index):
    if l==r:
        tree[treeNode]=value
        return
    mid=(l+r)//2
    if mid<index:
        update(tree,mid+1,r,2*treeNode+1,value,index)
    else:
        update(tree,l,mid,2*treeNode,value,index)

def updateInit(arr,tree,value,index):
    arr[index]=value
    update(tree,0,len(arr)-1,1,value,index)

def build(arr,tree,l,r,treeNode):
    if l==r:
        tree[treeNode]=arr[l]
        return
    mid=(l+r)//2
    build(arr,tree,l,mid,2*treeNode)
    build(arr,tree,mid+1,r,2*treeNode+1)
    tree[treeNode]=tree[2*treeNode]+tree[2*treeNode+1]

def query(tree,l,r,start,end,treeNode):
    if r<start or l>end:
        return 0
    if start<=l and r<=end:
        return tree[treeNode]
    mid=(l+r)//2
    ans1=query(tree,l,mid,start,end,2*treeNode)
    ans2=query(tree,mid+1,r,start,end,2*treeNode+1)
    return ans1+ans2

# arr=[1,2,3,4,5]
# tree=[0]*(4*len(arr))
# r=len(arr)-1
# build(arr,tree,0,r,1)
# update(tree,0,r,1,10,2)
# print(tree)
# print(query(tree,0,r,2,4,1))

class MaxPair:
    def __init__(self):
        self.max1=-sys.maxsize
        self.max2=-sys.maxsize

def buildTreeForMaxSumPair(arr,tree,l,r,treeNode):

    if l==r:
        node=MaxPair()
        node.max1=arr[l]
        node.max2=-sys.maxsize
        tree[treeNode]=node
        return
    mid=(l+r)//2
    buildTreeForMaxSumPair(arr,tree,l,mid,2*treeNode)
    buildTreeForMaxSumPair(arr,tree,mid+1,r,2*treeNode+1)
    node=MaxPair()
    node.max1=max(tree[2*treeNode].max1,tree[2*treeNode+1].max1)
    node.max2=min(max(tree[2*treeNode].max1,tree[2*treeNode+1].max2),
                  max(tree[2*treeNode].max2,tree[2*treeNode+1].max1))
    tree[treeNode]=node

def updateTreeForMaxSumPair(tree,l,r,treeNode,index,value):
    if l==r:
        node=MaxPair()
        node.max1=value
        node.max2=-sys.maxsize
        tree[treeNode]=node
        return
    mid=(l+r)//2
    if index > mid:
        updateTreeForMaxSumPair(tree, mid + 1, r, 2 * treeNode + 1, index, value)
    else:
        updateTreeForMaxSumPair(tree, l, mid, 2 * treeNode, index, value)
    tree[treeNode].max1 = max(tree[2 * treeNode].max1, tree[2 * treeNode + 1].max1)
    tree[treeNode].max2 = min(max(tree[2 * treeNode].max1, tree[2 * treeNode + 1].max2),
                              max(tree[2 * treeNode].max2, tree[2 * treeNode + 1].max1))

def queryTreeForMaxSumPair(tree,l,r,treeNode,start,end):
    if r<start or end<l:
        return MaxPair()
    if start<=l and end>=r:
        return tree[treeNode]
    mid=(l+r)//2
    ans1=queryTreeForMaxSumPair(tree,l,mid,2*treeNode,start,end)
    ans2=queryTreeForMaxSumPair(tree,mid+1,r,2*treeNode+1,start,end)
    node=MaxPair()
    node.max1=max(ans1.max1,ans2.max1)
    node.max2=min(max(ans1.max1,ans2.max2),max(ans1.max2,ans2.max1))
    return node

# arr=[2,3,1,5,7,6]
# tree=[None]*(3*len(arr))
# r=len(arr)-1
# buildTreeForMaxSumPair(arr,tree,0,r,1)
# updateTreeForMaxSumPair(tree,0,r,1,3,10)
# for node in tree[1:]:
#     if node:
#         print(node.max1,node.max2)
# ans=queryTreeForMaxSumPair(tree,0,r,1,1,3)
# print(ans.max1,ans.max2)

class MaxSubArraySum:
    def __init__(self):
        self.sum=0
        self.maxSum=0
        self.bestPrefSum=0
        self.bestSufSum=0

def buildTreeForMaxSubArray(arr,tree,l,r,treeNode):
    if l==r:
        node=MaxSubArraySum()
        node.sum=arr[l]
        node.maxSum=arr[l]
        node.bestPrefSum=arr[l]
        node.bestSufSum=arr[l]
        tree[treeNode]=node
        return
    mid=(l+r)//2
    buildTreeForMaxSubArray(arr,tree,l,mid,2*treeNode)
    buildTreeForMaxSubArray(arr,tree,mid+1,r,2*treeNode+1)
    node1=tree[2*treeNode]
    node2=tree[2*treeNode+1]
    node=MaxSubArraySum()
    node.sum=node1.sum +node2.sum
    node.bestPrefSum=max(node1.bestPrefSum,node1.sum+node2.bestPrefSum)
    node.bestSufSum=max(node2.bestSufSum,node2.sum+node1.bestSufSum)
    node.maxSum=max(node1.maxSum,node2.maxSum,node1.sum+node2.bestPrefSum,
                    node2.sum+node1.bestSufSum,node1.bestSufSum+node2.bestPrefSum)
    tree[treeNode]= node

def updateTreeForMaxSubArray(tree,l,r,treeNode,index,value):
    if l==r:
        node=MaxSubArraySum()
        node.sum=value
        node.maxSum=value
        node.bestPrefSum=value
        node.bestSufSum=value
        tree[treeNode]=node
        return
    mid=(l+r)//2
    if mid<index:
        updateTreeForMaxSubArray(tree,mid+1,r,2*treeNode+1,index,value)
    else:
        updateTreeForMaxSubArray(tree,l,mid,2*treeNode,index,value)
    n1=tree[2*treeNode]
    n2=tree[2*treeNode+1]
    node=MaxSubArraySum()
    node.sum=n1.sum+n2.sum
    node.bestPrefSum=max(n1.bestPrefSum,n1.sum+n2.bestPrefSum)
    node.bestSufSum=max(n2.bestSufSum,n2.sum+n1.bestSufSum)
    node.maxSum=max(n1.maxSum,n2.maxSum,n1.sum+n2.bestPrefSum,
                    n2.sum+n1.bestSufSum,n1.bestSufSum+n2.bestPrefSum)
    tree[treeNode]=node

def queryTreeForMaxSubArray(tree,l,r,treeNode,start,end):
    if r<start or end<l:
        return MaxSubArraySum()
    if start<=l and r<=end:
        return tree[treeNode]
    mid=(l+r)//2
    n1=queryTreeForMaxSubArray(tree,l,mid,2*treeNode,start,end)
    n2=queryTreeForMaxSubArray(tree,mid+1,r,2*treeNode+1,start,end)
    node=MaxSubArraySum()
    node.sum=n1.sum+n2.sum
    node.bestPrefSum=max(n1.bestPrefSum,n1.sum+n2.bestPrefSum)
    node.bestSufSum=max(n2.bestSufSum,n2.sum+n1.bestSufSum)
    node.maxSum=max(n1.maxSum,n2.maxSum,n1.sum+n2.bestPrefSum,
                    n2.sum+n1.bestSufSum,n1.bestSufSum+n2.bestPrefSum)
    return node


arr=[-2, -3, 4, -1, -2, 1, 5, -3]
tree=[None]*(3*len(arr))
r=len(arr)-1
buildTreeForMaxSubArray(arr,tree,0,r,1)
for node in tree:
    if node:
        print(node.sum,node.maxSum,node.bestPrefSum,node.bestSufSum)
node=queryTreeForMaxSubArray(tree,0,r,1,2,5)
print(node.sum,node.maxSum,node.bestPrefSum,node.bestSufSum)


