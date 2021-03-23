from collections import deque,defaultdict
from sys import maxsize
class Node :
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None

class BinaryTree :

    def __init__(self,root=None):
        self.root=Node(root)

def inOrder(root,traversal):

    if root:
        traversal=inOrder(root.left,traversal)
        traversal+=(str(root.data)+"-")
        traversal=inOrder(root.right,traversal)
    return traversal

def postOrder(root,traversal):

    if root:
        traversal=postOrder(root.left,traversal)
        traversal=postOrder(root.right,traversal)
        traversal+=(str(root.data)+"-")
    return traversal

def preOrder(root,traversal):

    if root:
        traversal+=(str(root.data)+"-")
        traversal=preOrder(root.left,traversal)
        traversal=preOrder(root.right,traversal)
    return traversal

def levelOrder(root,traversal):
    q=deque()
    q.append(root)
    while q:
        curr=q.popleft()
        traversal+=(str(curr.data)+"-")
        if curr.left:
            q.append(curr.left)
        if curr.right:
            q.append(curr.right)
# Starting///////////////////////////////////////////////////////////

def helperCounter(node,sum,count):
    if node is None:
        return 0
    lt=helperCounter(node.left,sum,count)
    rt=helperCounter(node.right,sum,count)
    curr_sum=lt+rt+node.data
    print(curr_sum)
    if curr_sum==sum:
        count[0]+=1
    return curr_sum

def countSubtreeWithSum(node,sum):
    if node is None:
        return 0
    count=[0]
    curr_sum=helperCounter(node,sum,count)
    return count[0]
# Ending////////////////////////////////////////////////////////////////////

# Staring///////////////////////////////////////////////////////////////////

def addInStack(node,s):
    curr=node
    while curr is not None:
        s.append(curr)
        curr=curr.left

def inOrderByStack(node):
    s=deque()
    addInStack(node,s)
    while s:
        curr=s.pop()
        print(curr.data)
        if curr.right:
            curr=curr.right
            while curr is not None:
                s.append(curr)
                curr=curr.left
# Ending//////////////////////////////////////////////////////////////////////////

def preOrderUsingStack(node):
    s=deque()
    s.append(node)
    while s:
        curr=s.pop()
        print(curr.data)
        if curr.right:
            s.append(curr.right)
        if curr.left:
            s.append(curr.left)

def postOrderUsing1Stack(node):
    s=deque()
    curr=node
    while curr or s:
        if curr:
            s.append(curr)
            curr=curr.left
        else:
            if s[-1].right :
                curr=s[-1].right
            else:
                temp=s.pop()
                print(temp.data)
                while s and s[-1].right is temp:
                    temp=s.pop()
                    print(temp.data)

# Starting /////////////////////////////////////////////////////////////////////////////

def diagonalPrintHelper(root,d,map):
    if not root:
        return
    map[d].append(root.data)
    diagonalPrintHelper(root.left,d+1,map)
    diagonalPrintHelper(root.right,d,map)

def diagonalPrint(root):
    map=defaultdict(list)
    diagonalPrintHelper(root,0,map)
    for i in map:
        for j in map[i]:
            print(j, end = ' ')
        print('')
# Ending ///////////////////////////////////////////////////////////////////////////////////
# Starting/////////////////////////////////////////////////////////////////////////////////
def verticalTraversalHelper(node,d,map):
    if not  node:
        return
    map[d].append(node.data)
    verticalTraversalHelper(node.left,d-1,map)
    verticalTraversalHelper(node.right,d+1,map)

def verticalTraversal(node):
    if not node:
        return
    map=defaultdict(list)
    verticalTraversalHelper(node,0,map)
    for i in map:
        for j in map[i]:
            print(j,end=" ")
        print("")
# Ending////////////////////////////////////////////////////////////////////////////////
# Starting/////////////////////////////////////////////////////////////////////////////////
def printLeftBoundary(root):
    if root:
        if root.left:
            print(root.data)
            printLeftBoundary(root.left)
        elif root.right:
            print(root.data)
            printLeftBoundary(root.right)

def printRightBoundary(root):
    if root:
        if root.right:
            printRightBoundary(root.right)
            print(root.data)
        elif root.left:
            printRightBoundary(root.left)
            print(root.data)

def printLeafNodes(root):
    if root:
        printLeafNodes(root.left)
        if not(root.left or root.right):
            print(root.data)
        printLeafNodes(root.right)

def printBoundaryNodes(root):
    if root:
        print(root.data)
        printLeftBoundary(root.left)
        printLeafNodes(root.left)
        printLeafNodes(root.right)
        printRightBoundary(root.right)
# Ending////////////////////////////////////////////////////////////////////
def printSpecialLevelOrder(root):
    if root:
        print(root.data)
    else:
        return
    q1=deque()
    if root.left:
        q1.append(root.left)
        q1.append(root.right)
    else:
        return

    while q1:
        l=len(q1)
        for _ in range(l//2):
            f=q1.popleft()
            s=q1.popleft()
            print(f.data,end=" ")
            print(s.data,end=" ")
            if f.left:
                q1.append(f.left)
                q1.append(s.right)
                q1.append(f.right)
                q1.append(s.left)
        print()
# Staring///////////////////////////////////////////////////////////////////
def sizeOfTree(root,size):
    if not root:
        return 0
    size[0]=sizeOfTree(root.left,size) + sizeOfTree(root.right,size) + 1
    return size[0]

def constructTree(pre,preLN):
    s=deque()
    root=Node(pre[0])
    s.append(root)

    if len(pre)<2:
        return root

    for i in range(1,len(pre)):
        curr=s.pop()
        if preLN[i] is "N":
            if curr.left is None:
                curr.left=Node(pre[i])
                s.append(curr)
                s.append(curr.left)
            elif curr.right is None:
                curr.right=Node(pre[i])
                s.append(curr.right)
        else:
            if curr.left is None:
                curr.left=Node(pre[i])
                s.append(curr)
            elif curr.right is None:
                curr.right=Node(pre[i])
    return root
# Ending//////////////////////////////////////////////////////////////////////

def createTreeFromParentArray(parent,n):
    created=[None for i in range(n)]
    root=None
    for i in range(n):
        created[i]=Node(i)
    for i in range(n):
        p=parent[i]
        if p is -1:
            root=created[i]
        else:
            node=created[p]
            if node.left:
                node.right=created[i]
            else:
                node.left=created[i]
    return root

# Starting///////////////////////////////////////////////////////////////
def createDLLUtil(root,head):
    if root is None:
        return head
    node=Node(root.data)
    if head:
        head.next=node
        node.prev=head

    lend=createDLLUtil(root.left,node)
    mend=createDLLUtil(root.mid,lend)
    rend=createDLLUtil(root.right,mend)
    return rend

def createDLL(root):

    curr=createDLLUtil(root,None)
    while curr.prev :
        print(curr.data,end=" ")
        curr=curr.prev
    print(curr.data,end=" ")
    return curr
# Ending////////////////////////////////////////////////////////////////
# Starting///////////////////////////////////////////////////////
def createDllFromBinaryTreeUtil(root,head):

    if root is None:
        return head
    lend=createDllFromBinaryTreeUtil(root.left , head)
    node=Node(root.data)
    if lend:
        lend.right=node
        node.left=lend
    rightend=createDllFromBinaryTreeUtil(root.right,node)
    return rightend

def createDllFromBinaryTree(root):
    curr=createDllFromBinaryTreeUtil(root,None)
    while curr.left:
        curr=curr.right
    return curr
# Ending////////////////////////////////////////////////////////
def convertIntoSumTree(root):
    if root is None:
        return 0
    lsum=sum(root.left)
    rsum=sum(root.right)
    result=root.data+lsum+rsum
    root.data=result-root.data
    return result
# Starting////////////////////////////////////////////////////////
def createCdllUtil(root,head):
    if root is None:
        return head
    lend=createCdllUtil(root.left,head)
    node=Node(root.data)
    if lend:
        lend.right=node
        node.left=lend
    rend=createCdllUtil(root.right,node)
    return rend

def createCdll(root):
    if root is None:
        return root
    end=createCdllUtil(root,None)
    curr=end
    while curr.left:
        curr=curr.left
    end.right=curr
    curr.left=end
    return curr
# Ending////////////////////////////////////////////////////////////////
# Starting///////////////////////////////////////////////////////
def findPath(root,element,path):
    if not root:
        return 0
    path.append(root.data)
    if root.data is element or findPath(root.left,element,path) or findPath(root.right,element,path):
        return 1
    else:
        path.pop()
        return 0

def findAncestor(root,n1,n2):
    p1=deque()
    findPath(root,n1,p1)
    print(p1)
    p2=deque()
    findPath(root,n2,p2)
    ans=None
    while p1 and p2:
        if p1[0] is p2[0]:
            p1.pop()
            ans=p2.pop()
        else:
            break
    return ans
# Ending////////////////////////////////////////////////////////////////

def LCAUtil(root,n1,n2):

    if not root:
        return None
    if root.data is n1 or root.data is n2:
        return root

    lst=LCAUtil(root.left,n1,n2)
    rst=LCAUtil(root.right,n1,n2)
    if lst and rst:
        return root
    if lst:
        return lst
    else:
        return rst

def inOrderPrint(root,parent):
    if root is None:
        return None
    lst=inOrderPrint(root.left,parent)

    if lst:
        lst.next=root
    elif parent:
        parent.next=root
    if root.right:
        return inOrderPrint(root.right,root)
    else:
        return root

def connectSameLevelNode(root):
    pl=root
    cl=None
    start=None
    while True:
        while pl:
            if pl.left:
                if cl:
                    cl.nextRight=pl.left
                    cl=cl.nextRight
                else:
                    cl=pl.left
                    start=pl.left
            if pl.right:
                if cl:
                    cl.nextRight=pl.right
                    cl=cl.nextRight
                else:
                    cl=pl.right
                    start=pl.right
            pl=pl.nextRight
        if cl:
            pl=start
            cl=None
        else:
            break
# Starting///////////////////////////////////////////////////////
def foldUtil(st1,st2):
    if ((not st1)and st2)or((not st2)and st1):
        if st1:
            print(st1.data)
        if st2:
            print(st2.data)
        return False
    if st1 and st2:
        # print(st1.data,st2.data)
        if foldUtil(st1.left,st2.right) and foldUtil(st1.right,st2.left):
            return True
        else:
            print(st1.data,st2.data)
            return False
    else:
        return True


def fold(root):
    return foldUtil(root.left,root.right)
# Ending///////////////////////////////////////////////////////////////
# Starting////////////////////////////////////////////////
# To Find the max width in a tree by different approaches

# Approach 1: Using recursion
def  height(root):
    if not root:
        return 0
    return 1+ max(height(root.left),height(root.right))

def findWidthUtil(root,level):
    if root is None:
        return 0
    if level is 1:
        return 1
    return findWidthUtil(root.left,level-1) + findWidthUtil(root.right,level-1)

def findWidth(root):
    h=height(root)
    w=0
    for i in range(1,h+1):
        w=max(findWidthUtil(root,i),w)
    return w

# Approach 2: Using queue
def  findWidthUsingQueue(root):
    w=0
    q=deque()
    q.append(root)
    while q:
        l=len(q)
        w=max(w,l)
        for i in range(l):
            curr=q.popleft()
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
    return w
# Ending//////////////////////////////////////////////////
def duplicate(root):
    if not root:
        return None
    node=Node(root.data)
    l=root.left
    root.left=node
    node.left=l
    duplicate(l)
    duplicate(root.right)
    return
# Starting///////////////////////////////////////
def findPathWithSum(root,sum):
    if root is None:
        return False
    if (sum>0 and (findPath(root.left,sum-root.data)or findPath(root.right,sum-root.data))) or sum==root.data:
        print(root.data)
        return True
    return False

def maxSum(root):
    if not root:
        return 0
    return root.data + max(maxSum(root.left),maxSum(root.right))

def maxSumWPath(root):
    if not root:
        return None
    max_sum=maxSum(root)
    findPathWithSum(root,max_sum)
# Ending///////////////////////////////////////////////
# Starting//////////////////////////////////////////////
def verticalSumUtil(root,map,level):
    if not root:
        return None
    map[level]+=root.data
    verticalSumUtil(root.left,map,level-1)
    verticalSumUtil(root.right,map,level+1)

def verticalSum(root):
    if root is None:
        return 0
    map=defaultdict(int)
    verticalSumUtil(root,map,0)
    ans=[]
    for i in sorted(map):
        ans.append(map[i])
    return ans
# Ending//////////////////////////////////////////////
def nextRight(root,num):
    if not root:
        return None
    q=deque()
    q.append(root)
    while q:
        l=len(q)
        for i in range(l):
            curr=q.popleft()
            if curr.data==num:
                if i+1<l:
                    return q.popleft().data
                else:
                    return None
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
    return None
# Starting////////////////////////////////////////////
def deepLeftLeafUtil(root,ans,level):
    if not root:
        return None
    deepLeftLeafUtil(root.left,ans,level+1)
    if root.left:
        if ans[0]<level+1:
            ans[0]=level+1
            ans[1]=root.left.data
    deepLeftLeafUtil(root.right,ans,level+1)

def deepLeftLeaf(root):
    if not root:
        return None
    a=[0,None]
    deepLeftLeafUtil(root,a,0)
    return a[1]
# Ending/////////////////////////////////////////////
# Starting/////////////////////////////////////////
def createDllUtil(root,node,p):
    if not root:
        return node
    if root.left is None and root.right is None:
        n=Node(root.data)
        if node:
            node.right=n
            n.left=node
            node=node.right
        else:
            node=n
        if p.left and p.left.data== root.data:
            p.left=None
        elif p.right and p.right.data==root.data:
            p.right=None
        return node
    else:
        new_node=createDllUtil(root.left,node,root)
        return createDllUtil(root.right,new_node,root)

def createDllOfLeaf(root):
    if root is None:
        return None
    end=createDllUtil(root,None,None)
    while end.left:
        end=end.left
    return end
# Ending//////////////////////////////////////////////
def leftView(root):
    if root is None:
        return None
    ans=deque()
    q=deque()
    q.append(root)
    while q:
        l=len(q)
        ans.append(q[0].data)
        for i in range(l):
            curr=q.popleft()
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
    return ans

def alternateLevelReverse(root):
    if root is None:
        return None
    q=deque()
    q.append(root)
    i=1
    while q:
        l=len(q)
        if i%2==0:
            j=0
            k=l-1
            while j<k:
                temp=q[j].data
                q[j].data=q[k].data
                q[k].data=temp
                j+=1
                k-=1
        for _ in range(l):
            curr=q.popleft()
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
        i+=1
# Starting/////////////////////////////////////////////////
def findPath2(root,node,path):
    if root is None:
        return False
    if root.data== node:
        path.append(root.data)
        return True
    elif findPath(root.left,node,path) or findPath(root.right,node,path):
        path.append(root.data)
        return True
    else:
        return False


def nodeAtKUtil(root,path,dist,k,ans):

    if root is None:
        return

    if dist==k:
        print(root.data)
        ans.append(root.data)

    if root.left:
        if root.left.data in path:
            nodeAtKUtil(root.left,path,dist-1,k,ans)
        elif dist<k:
            nodeAtKUtil(root.left,path,dist+1,k,ans)

    if root.right:
        if root.right.data in path:
            nodeAtKUtil(root.right,path,dist-1,k,ans)
        elif dist<k:
            nodeAtKUtil(root.right,path,dist+1,k,ans)

    return
def KDistanceNodes(root,target,k):
    if root is None:
        return
    path=[]
    findPath2(root,target,path)
    ans=[]
    nodeAtKUtil(root,path,len(path)-1,k,ans)
    return sorted(ans)
# Ending///////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def findLen(root,n1):
    if root is None:
        return 0
    lst=findLen(root.left,n1)
    rst=findLen(root.right,n1)
    if root.data==n1:
        return 1
    if lst!=0 or rst!=0:
        return 1+lst+rst
    else:
        return 0

def LCA(root,n1,n2):
    if root is None:
        return None
    if root.data== n1 or root.data== n2:
        return root
    lst=LCA(root.left,n1,n2)
    rst=LCA(root.right,n1,n2)
    if lst and rst:
        return root
    if lst:
        return lst
    else:
        return rst

def findDistBtwNodes(root,n1,n2):
    anc=LCA(root,n1,n2)
    return findLen(anc,n1)+findLen(anc,n2)-2
# Ending//////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def printDiagonal(root,map,level):
    if root is None:
        return None

    map[level]+=root.data
    printDiagonal(root.left,map,level+1)
    printDiagonal(root.right,map,level)

def printDiagonalUtil(root):
    map=defaultdict(int)
    printDiagonal(root,map,0)
    return map
# Ending//////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def  tilt(root,ans):
    if root is None:
        return 0
    lst=tilt(root.left,ans)
    rst=tilt(root.right,ans)
    ans[0]+=abs(lst-rst)
    return root.data+lst+rst
def tiltUtil(root):
    ans=[0]
    tilt(root,ans)
    return ans[0]
# Ending//////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def findDepth(root,map,arr):
    if arr[root]==-1:
        return 1
    if map[arr[root]]!=0:
        map[root]=1+map[arr[root]]
    else:
        map[root]=1+findDepth(arr[root],map,arr)
    return map[root]

def findMaxDepthUtil(arr):
    map=defaultdict(int)
    depth=0
    for i in range(len(arr)):
        if map[i]!=0:
            depth=max(map[i],depth)
        else:
            depth=max(findDepth(i,map,arr),depth)
    return depth
# Ending//////////////////////////////////////////////
def removeShortPathNodesUtil(root,k, parent):
    if root is None:
        return False
    if k<=1:
        return True
    lst=removeShortPathNodesUtil(root.left,k-1,root)
    rst=removeShortPathNodesUtil(root.right,k-1,root)
    if (not lst) and  (not rst):
        if parent:
            if parent.left is root:
                parent.left=None
            else:
                parent.right=None
            return False
    return True

def maxPathSum(root,sum):
    if root is None:
        return 0
    lst=maxPathSum(root.left,sum)
    rst=maxPathSum(root.right,sum)
    curr=root.data+lst+rst
    if  curr>sum[0]:
        sum[0]=curr
    return root.data+max(lst,rst)

def maxPathSumUtil(root):
    sum=[0]
    ans=maxPathSum(root,sum)
    return max(ans,sum[0])

# Starting///////////////////////////////////////////////
def infixToPostfix(string):
    list=[]
    s=deque()
    d={"/":2,"*":2,"+":1,"-":1,"^":3}
    for ch in string:
        if ch.isalpha():
            list.append(ch)
        elif ch in d:
            if s:
                curr=s[-1]
                while s and curr is not "(" and d[curr]>=d[ch]:
                    list.append(s.pop())
                    if s:
                        curr=s[-1]
                    else:
                        break
            s.append(ch)
        elif ch  is "(":
            s.append(ch)
        elif ch is ")":
            while s and s[-1] is not "(":
                list.append(s.pop())
            s.pop()
    while s:
        list.append(s.pop())
    return list

def postfixToInfix(string):
    s=deque()
    ans=""
    for ch in string:
        node=Node(ch)
        if not ch.isalpha():
            node.right=s.pop()
            node.left=s.pop()
        s.append(node)

    return s.pop()
# Ending//////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def findPath3(root,node,path):
    if root  is None:
        return False
    if root.data is node:
        path.append(root)
        return True
    elif findPath(root.left,node,path) or findPath(root.right,node,path):
        path.append(root)
        return True
    return False
def reversePath(root,node):
    path=[]
    findPath3(root,node,path)
    i=0
    j=len(path)-1
    while i<j:
        temp=path[i].data
        path[i].data=path[j].data
        path[j].data=temp
        i+=1
        j-=1
# Ending//////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def findMax3(root,sum):
    if root is None:
        return -maxsize
    lst=findMax3(root.left,sum)
    rst=findMax3(root.right,sum)
    sum[0]=max(sum[0],lst+rst+root.data)
    large=max(lst,rst)
    if large>(-maxsize):
        return root.data+large
    else:
        return root.data

def findMaxSumBtwTwoNodes(root):
    sum=[-maxsize]
    curr=[0]
    ans=findMax3(root,sum)
    return sum[0]

# Ending//////////////////////////////////////////////
# Starting///////////////////////////////////////////////
def serialize(root):
    arr=[]
    q=deque()
    q.append(root)
    while q:
        curr=q.popleft()
        if curr=="N":
            arr.append("N")
        else:
            arr.append(curr.data)
            if curr.left:
                q.append(curr.left)
            else:
                q.append("N")
            if curr.right:
                q.append(curr.right)
            else:
                q.append("N")
    return arr

def deserialize(arr):
    l=len(arr)
    i=1
    q=deque()
    root=Node(arr[0])
    q.append(root)
    while i<l and q:
        curr=q.popleft()
        if arr[i] is not "N":
            curr.left=Node(arr[i])
            q.append(curr.left)
        i+=1
        if arr[i] is not "N":
            curr.right=Node(arr[i])
            q.append(curr.right)
        i+=1
    return root
# Ending//////////////////////////////////////////////

def inorderSuccessor(root,ans,key):

    if root is None:
        return False
    if root.data == key :
        if root.right:
            curr=root.right
            while curr.left:
                curr=curr.left
            ans[0]=curr.data
        return True
    lst=inorderSuccessor(root.left,ans,key)
    if lst is True:
        if ans[0] is None:
            ans[0]=root.data
        return True
    else:
        return inorderSuccessor(root.right,ans,key)

# Set up tree:
tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)
tree.root.right.left = Node(6)
tree.root.right.right = Node(7)

print(preOrder(tree.root,""))

