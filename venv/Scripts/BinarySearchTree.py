from collections import deque
class Node:
    def __init__(self, data):
        self.data = data
        self.left = self.right = None

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

# Starting////////////////////////////////////////////
# Return the root of the modified BST after deleting the node with value X
def inorderSuccessor2(root,ans,key,parent,node):
    if root is None:
        return False
    currParent=parent[0]
    parent[0]=root
    if root.data == key :
        node[0]=root
        if root.right:
            curr=root.right
            while curr.left:
                parent[0]=curr
                curr=curr.left
            ans[0]=curr.data
        return True
    lst=inorderSuccessor2(root.left,ans,key,parent,node)
    if lst is True:
        if ans[0] is None:
            parent[0]=currParent
            ans[0]=root.data
        return True
    else:
        parent[0]=root
        return inorderSuccessor2(root.right,ans,key,parent,node)

def deleteNode(root,key):
    if root.data==key and root.right is None and root.left is None:
        return None

    ans=[None]
    node=[None]
    parent=[None]
    inorderSuccessor2(root,ans,key,parent,node)
    if node[0] is not None:
        node[0].data=ans[0]
        if parent[0].left and parent[0].left.data==ans:
            parent[0].left=None
        else:
            parent[0].right=None
    return root
# Ending///////////////////////////////////////////////////////
# Return the root of the modified BST after deleting the node with value X
def isBST(root,low,high):
    if root is None:
        return True
    val=root.data
    if val <=low or val >=high:
        return False
    if not isBST(root.left,low,val):
        return False
    if not isBST(root.right,val,high):
        return False
    return True

def isBstHelper(root):
    return isBST(root,-maxsize,maxsize)
# Ending///////////////////////////////////////////////////////
# Ending///////////////////////////////////////////////////////
def createTreeFromPreorder(root,lower,higher,i,arr):
    if root is None or i[0]>=len(arr):
        return
    if arr[i[0]]>lower and arr[i[0]]<root.data:
        root.left=Node(arr[i[0]])
        i[0]+=1
        createTreeFromPreorder(root.left,lower,root.data,i,arr)
    if arr[i[0]]>root.data and arr[i[0]]<higher:
        root.right=Node(arr[i[0]])
        i[0]+=1
        createTreeFromPreorder(root.right,root.data,higher,i,arr)

def createTreeUtil(arr):
    if len(arr)>0:
        root=Node(arr[0])
        i=[1]
        createTreeFromPreorder(root,-maxsize,maxsize,i,arr)
        return root
    return None
# Ending///////////////////////////////////////////////////////
# Starting//////////////////////////////////
def inOrder(root,arr):
    if root is None:
        return
    inOrder(root.left,arr)
    arr.append(root.data)
    inOrder(root.right,arr)
    return

def convetIntoBST(root,arr,i):
    if root is None or i[0]>=len(arr):
        return
    convetIntoBST(root.left,arr,i)
    root.data=arr[i[0]]
    i[0]+=1
    convetIntoBST(root.right,arr,i)

def convertBinaryTreeIntoBst(root):
    arr=[]
    inOrder(root,arr)
    print(arr)
    i=[0]
    convetIntoBST(root,sorted(arr),i)
# Ending///////////////////////////////////////////////////////
# Starting/////////////////////////////////////////
def LCA(root,n1,n2):

    if root is None:
        return root
    if root.data is n1 or root.data is n2:
        return root.data
    if root.data >n1 and root.data<n2:
        return root.data
    if root.data>n1:
        return LCA(root.left,n1,n2)
    if root.data<n2:
        return LCA(root.right,n1,n2)

def LCAUtil(root,n1,n2):
    if n1>n2:
        temp=n1
        n1=n2
        n2=temp
    return LCA(root,n1,n2)
# Ending///////////////////////////////////////////////////////
# Starting/////////////////////////////////////////
def intersectionOfll(l1,l2):
    ans=[]
    while l1 and l2:
        if l1.data==l2.data:
            ans.append(l1.data)
            l1=l1.right
            l2=l2.right
        elif l1.data<l2.data:
            l1=l1.right
        elif l2.data<l1.data:
            l2=l2.right
    return ans

def  BstIntoll(root,head):
    if root is None:
        return None
    BstIntoll(root.right,head)
    root.right=head[0]
    if head[0]:
        head[0].left=None
    head[0]=root
    BstIntoll(root.left,head)

def intersectionUtil(r1,r2):
    head1=[None]
    head2=[None]
    r1=BstIntoll(r1,head1)
    r2=BstIntoll(r2,head2)
    return intersectionOfll(head1[0],head2[0])
# Ending///////////////////////////////////////////////////////
# Starting//////////////////////////////////////////
def printNode(root,min,max,ans):
    if root is None:
        return None
    if not root.data<min:
        printNode(root.left,min,max,ans)
    if root.data>=min and root.data<=max:
        ans.append(root.data)
    if not root.data>max:
        printNode(root.right,min,max,ans)

def printNodeInRangeUtil(root,min,max):
    ans=[]
    printNode(root,min,max,ans)
    return ans
#Ending///////////////////////////////////////////////////
# Starting///////////////////
def findLen(root):
    if root is None:
        return 0
    return 1+findLen(root.left)+findLen(root.right)

def median(root,level,i,ans):
    if root is None:
        return None

    median(root.left,level,i,ans)
    level[0]+=1
    if i%2==0:

        if level[0]==i//2+1 or level[0]==i//2:
            # print(level,root.data)
            ans[0]+=root.data
    else:
        if level[0]==(i//2+1):
            ans[0]+=root.data
    print(level[0],root.data)
    median(root.right,level,i,ans)

def findMedian(root):
    l=findLen(root)
    ans=[0]
    level=[0]

    median(root,level,l,ans)
    print(ans)
    if l%2==0:
        if ans[0]%2:
            return ans[0]//2
        else:
            return ans[0]/2
    else:
        return ans[0]
# ending///////////////////////////////////////
# Starting//////////////////////////////
def Klargest(root,level,i,ans):
    if root is None:
        return None
    Klargest(root.right,level,i,ans)
    level[0]+=1
    if level[0]==i:
        ans[0]=root.data
        return
    Klargest(root.left,level,i,ans)

def KlargestUtil(root,k):
    level=[0]
    ans=[0]
    Klargest(root,level,k,ans)
    return ans[0]
# Ending//////////////////////////////////
# Starting////////////////////////////////
def BstToList(root,arr):
    if root is None:
        return
    BstToList(root.left,arr)
    arr.append(root.data)
    BstToList(root.right,arr)

def buildCompleteBstFromlist(arr,start,end):
    if start>end:
        return None
    i=(start+end)//2
    node=Node(arr[i-1])
    node.left=buildCompleteBstFromlist(arr,start,i-1)
    node.right=buildCompleteBstFromlist(arr,i+1,end)
    return node
def BstToCompleteBst(root):
    ans=[]
    BstToList(root,ans)
    root=buildCompleteBstFromlist(ans,1,len(ans))
    return root

# Ending//////////////////////////////////////////////
# Starting////////////////////////////////
def buildBst(root,x):
    if root.data>x:
        if root.left:
            buildBst(root.left,x)
        else:
            root.left=Node(x)
    else:
        if root.right:
            buildBst(root.right,x)
        else:
            root.right=Node(x)

def fromPostorderintoBst(arr,size):
    root=Node(arr[-1])
    for x in arr[size-2::-1]:
        buildBst(root,x)
    return root
# Ending//////////////////////////////////////////////

def checkHeap(root):
    q=deque()
    q.append(root)
    flag=0
    while q and flag==0:
        l=len(q)
        for i in range(l):
            curr=q.popleft()
            if curr is not "N":
                if curr.left:
                    if curr.left.data<curr.data:
                        q.append(curr.left)
                    else:
                        flag=1
                        break
                else:
                    q.append("N")
                if curr.right:
                    if curr.right.data<curr.data:
                        q.append(curr.right)
                    else:
                        flag=1
                        break
                else:
                    q.append("N")
            else:
                # print(i,l)
                while i<l-1:
                    if not curr=="N":
                        flag=1
                        break
                    curr=q.popleft()
                    i+=1
                break
    if flag==1:
        print(0)
    else:
        print(1)
# Starting////////////////////////////////
def find(root,k):
    if root is None:
        return 0
    if root.data == k:
        return 1
    if root.data>k:
        return find(root.left,k)
    else:
        return find(root.right,k)

def findBrother(root,sum,ans,root2):
    if root is None:
        return
    ans[0]+=find(root2,sum-root.data)
    findBrother(root.left,sum,ans,root2)
    findBrother(root.right,sum,ans,root2)

def countPairs(root1,root2,k):
    ans=[0]
    findBrother(root1,k,ans,root2)
    return ans[0]
# Ending//////////////////////////////////////
def findDeadEnd(root,lower,higher):
    if root is None:
        return 0
    print(root.data,lower,higher)
    if higher-lower==2:
        return 1
    return findDeadEnd(root.left,lower,root.data) or findDeadEnd(root.right,root.data,higher)
# Starting/////////////////////////////
def findSum(first,sum,n1,n2):
    if n1 is n2 or  n1.data>n2.data:
        return 0
    currsum=n1.data + n2.data
    if currsum == sum:
        return 1
    elif currsum >sum:
        return findSum(first,sum,n1,n2.left)
    else:
        return findSum(first,sum,n1.right,n2)

def  BstIntoDll(root,head):
    if root is None:
        return head
    rst=BstIntoDll(root.right,head)
    if rst is not None:
        root.right=rst
        rst.left=root
    return BstIntoDll(root.left,root)


def findTriplet(head):
    curr=head
    start=None
    end=None
    while curr:
        if start is None and curr.data>0:
            start=curr
        end=curr
        curr=curr.right
    # print(start.data,end.data)
    while head and head.data<0:
        if findSum(head,-1*head.data,head.right,end):
            return 1
        head=head.right
    return 0
# Ending/////////////////////////////////////
def levelOrderToBst(arr):
    root=Node(arr[0])
    q=deque()
    q.append([root,-maxsize,maxsize])
    i=1
    l=len(arr)
    while i<l and q:
        curr=q.popleft()
        node=curr[0]
        if arr[i]>curr[1] and arr[i]<curr[2]:
            new_node=Node(arr[i])
            if arr[i]<node.data:
                node.left=new_node
                q.append([new_node,curr[1],node.data])
            else:
                node.right=new_node
                q.append([new_node,node.data,curr[2]])
            i+=1
        if i<l and node.right==None:
            if arr[i]>curr[1] and arr[i]<curr[2]:
                new_node=Node(arr[i])
                if arr[i]>node.data:
                    node.right=new_node
                    q.append([new_node,node.data,curr[2]])
                i+=1
    return root

# Starting///////////////////////////////////////////////
def isIdentical(a,b,n,i,j,min,max):
    while i<n:
        if a[i]>min and a[i]<max:
            break
        i+=1
    while j<n:
        if b[j]>min and b[j]<max:
            break
        j+=1
    if i==n and j==n :
        return True
    if (i==n)^(j==n) or not(a[i]==b[j]):
        return False
    return isIdentical(a,b,n,i+1,j+1,a[i],max) and isIdentical(a,b,n,i+1,j+1,min,a[i])

def isIdenticalUtil(a,b,size):
    return isIdentical(a,b,size,0,0,-maxsize,maxsize)
# Ending/////////////////////////////
def morriseTraversalKthSmallest(root,k):
    curr=root
    count=1
    while curr:
        if curr.left is None:
            if count==k:
                print(curr.data)
                break
            else:
                count+=1
                curr=curr.right
        else:
            prev=curr.left
            # print(prev.data)
            while prev.right and prev.right !=curr:
                # print(prev.data)
                prev=prev.right
            if prev.right:
                if k==count:
                    print(curr.data)
                    break
                else:
                    count+=1
                    prev.right=None
                    curr=curr.right
            else:
                prev.right=curr
                curr=curr.left

def createBstFromLL(head,n):
    if n==0 or head==None:
        return None
    lstSize=n//2
    lst=createBstFromLL(head,lstSize)
    root=Node(head[0].data)
    head[0]=head[0].right
    root.left=lst
    root.right=createBstFromLL(head,n-lstSize-1)
    return root
# Starting//////////////////////////////
def convertBstIntoDll(root,head):
    if root is None:
        return None
    convertBstIntoDll(root.right,head)
    root.right=head[0]
    if head[0]:
        head[0].left=root
    head[0]=root
    convertBstIntoDll(root.left,head)

def mergerSortedDll(head1,head2):
    if head1 is None:
        return head2
    if head2 is None:
        return head1
    if head1.data>head2.data:
        curr=head2
        head=mergerSortedDll(head1,head2.right)
    else:
        curr=head1
        head=mergerSortedDll(head1.right,head2)
    curr.right=head
    head.left=curr
    return curr

def convertDllIntoBst(head,n):
    if  n==0 or  head is None:
        return None
    lstSize=n//2
    lst=convertDllIntoBst(head,lstSize)
    root=Node(head[0].data)
    head[0]=head[0].right
    root.left=lst
    root.right=convertDllIntoBst(head,n-lstSize-1)
    return root

def findSize(head):
    size=0
    curr=head
    while curr:
        size+=1
        curr=curr.right
    return size

def mergeTwoBst(root1,root2):
    if root1 is None:
        return root2
    if root2 is None:
        return root1
    head1=[None]
    convertBstIntoDll(root1,head1)
    head2=[None]
    convertBstIntoDll(root2,head2)
    head=mergerSortedDll(head1[0],head2[0])
    while head.left:
        head=head.left
    return convertDllIntoBst([head],findSize(head))
# Ending///////////////////////////////////////////
# Starting//////////////////////////////////////////////
def inOrder1(root,node1,node2,prev):
    if root is None:
        return
    inOrder(root.left,node1,node2,prev)
    if prev[0] and prev[0].data>root.data:
        if node1[0]:
            node2[0]=root
        else:
            node1[0]=prev[0]
            node2[0]=root
    prev[0]=root
    inOrder(root.right,node1,node2,prev)

def swapTwoInvalidNodeInBst(root):
    node1=[None]
    node2=[None]
    inOrder1(root,node1,node2,[None])
    if node1[0]and node2[0]:
        print(node1[0].data,node2[0].data)
        temp=node1[0].data
        node1[0].data=node2[0].data
        node2[0].data=temp
# Ending////////////////////////////////////////
# Starting/////////////////////////////////////////
def convertBstIntoDll(root,head):
    if root is None:
        return None
    convertBstIntoDll(root.right,head)
    root.right=head[0]
    if head[0]:
        head[0].left=root
    head[0]=root
    convertBstIntoDll(root.left,head)

def findSum(head,sum):
    end=head
    while end.right:
        end=end.right
    # print(head.data,end.data)
    while head.data<end.data:
        currSum=head.data+end.data
        # print(head.data,end.data)
        if currSum==sum:
            return True
        elif currSum>sum:
            end=end.left
        else:
            head=head.right
    return False

def findSumInBst(root,sum):
    head=[None]
    convertBstIntoDll(root,head)
    return findSum(head[0],sum)
# Ending//////////////////////////////////
def inOrderUsingStack(root):
    s=deque()
    curr=root
    while True:
        if curr:
            while curr:
                s.append(curr)
                curr=curr.left
        elif s:
            curr=s.pop()
            print(curr.data,end=" ")
            curr=curr.right
        else:
            break

def removeNodeOfGivenRange(root,min,max,parent):
    if root is None:
        return
    if root.data>min and root.data<max:
        removeNodeOfGivenRange(root.left,min,max,root)
        removeNodeOfGivenRange(root.right,min,max,root)
    elif root.data<min:
        if parent:
            if parent.left and parent.left.data==root.data:
                parent.left=root.right
            else:
                parent.right=root.right
        removeNodeOfGivenRange(root.right,min,max,parent)
    elif root.data>max:
        if parent:
            if parent.left and parent.left.data==root.data:
                parent.left=root.left
            else:
                parent.right=root.left
        removeNodeOfGivenRange(root.left,min,max,parent)

# Starting//////////////////////////////
def replace(root,sum):
    if root is None:
        return
    replace(root.right,sum)
    root.data+=sum[0]
    sum[0]=root.data
    replace(root.left,sum)

def replaceUtil(root):
    replace(root2,[0])

# Ending/////////////////////////
def findInorderPredecessorSuccessor(root,prev,s,p,key):
    if root is None:
        return
    if key<root.data:
        findInorderPredecessorSuccessor(root.left,prev,s,p,key)
    elif root.data==key:
        p[0]=prev[0]
    if prev[0] and prev[0].data==key:
        s[0]=root
    prev[0]=root
    if key>root.data:
        findInorderPredecessorSuccessor(root.right,prev,s,p,key)

def countBstInGivenRange(root, min, max, count):
    if root is None:
        return True
    lst=countBstInGivenRange(root.left, min, max, count)
    rst=countBstInGivenRange(root.right, min, max, count)
    if lst and rst :
        if root.data>min and root.data<max:
            count[0]+=1
            return True
    return False

# Starting////////////////////////

def getMinimumDifferenceHelper(root, pre):
    if root is None:
        return maxsize
    leftMin=getMinimumDifferenceHelper(root.left,pre)
    currMin=maxsize
    if pre[0] is not 0:
        currMin=abs(pre[0]-root.data)
    # print(root.data,pre)
    pre[0]=root.data
    rightMin=getMinimumDifferenceHelper(root.right,pre)
    return min(leftMin,currMin,rightMin)


def getMinimumDifference(root):
    pre=[0]
    return getMinimumDifferenceHelper(root,pre)
# Ending/////////////////////////

def trimRoot(root,low,high):
    if root is None:
        return root
    if root.data >= low and root.data <= high:
        root.left=trimRoot(root.left,low,high)
        root.right=trimRoot(root.right,low,high)
        return root
    else:
        if root.data<low:
            # root.left=None
            return trimRoot(root.right,low, high)
        else:
            # root.right=None
            return trimRoot(root.left,low,high)

def isCousins(root,x,y):
    if root is None:
        return False
    q=deque()
    flagx=True
    flagy=True
    q.append(root)
    while q:
        size=len(q)
        for _ in range(size):
            curr=q.popleft()
            if curr.left:
                if flagx and x is curr.left.data:
                    if curr.right and y==curr.right.data:
                        return False
                    flagx=False
                if flagy and y is curr.left.data:
                    if curr.right and x==curr.right.data:
                        return False
                    flagy=False
                q.append(curr.left)
            if curr.right:
                if flagx and x is curr.right.data:
                    flagx=False
                if flagy and y is curr.right.data:
                    flagy=False
                q.append(curr.right)
        if (not flagx) and (not flagy):
            return True
        if (not flagx) or (not flagy):
            return False
    return False

# Starting////////////////////////
