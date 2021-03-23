import sys
from collections import defaultdict
class trieNode:
    def __init__(self):
        self.left=None
        self.right=None
        self.weight=0

def insert(head,num):
    curr=head
    for i in range(31,-1,-1):
        bit=(num>>i)&1
        if bit:
            if not curr.right:
                curr.right=trieNode()
            curr=curr.right
        else:
            if not curr.left:
                curr.left=trieNode()
            curr=curr.left

def findMaxXorPair(head,arr,n):
    ans=-sys.maxsize
    for num in arr:
        currXor=0
        curr=head
        for i in range(31,-1,-1):
            bit= (num>>i)&1
            if bit==1:
                if curr.left:
                    currXor+=(1<<i)
                    curr=curr.left
                else:
                    curr=curr.right
            else:
                if curr.right:
                    currXor+=(1<<i)
                    curr=curr.right
                else:
                    curr=curr.left
        if currXor>ans:
            ans=currXor
    return ans

def subarrayOfMaxXor(arr):
    head=trieNode()
    xorProduct=arr[0]
    insert(head,xorProduct)
    maxXor=xorProduct
    for num in arr[1:]:
        xorProduct^=num
        insert(head,xorProduct)
        curr=head
        currXor=0
        for i in range(31,-1,-1):
            bit=(xorProduct>>i)&1
            if bit:
                if curr.left:
                    currXor+=(1<<i)
                    curr=curr.left
                else:
                    curr=curr.right
            else:
                if curr.right:
                    currXor+=(1<<i)
                    curr=curr.right
                else:
                    curr=curr.left
        if currXor>maxXor:
            maxXor=currXor
    print(maxXor)
#/////////////////////////////////////////////////////////////
class StringTrieNode:
    def __init__(self):
        self.children=defaultdict()
        self.weight=0

def insertForFindSubStringWithMaxWeight(string,weight,root):
    if len(string)==0:
        return
    child=root.children.get(string[0],None)
    if child:
        root=child
    else:
        node=StringTrieNode()
        root.children[string[0]]=node
        root=node
    if root.weight<weight:
        root.weight=weight
    insertForFindSubStringWithMaxWeight(string[1:],weight,root)


def findSubStringWithMaxWeight(head,subStr):
    bestWt=0
    for char in subStr:
        child=head.children.get(char,None)
        if child:
            head=child
            bestWt=child.weight
        else:
            return -1
    return bestWt
# strings=["bat","ben","ball","brian","alex","ford"]
# weight=[5,10,20,15,31,24]
# head=StringTrieNode()
# for i in range(len(strings)):
#     insertForFindSubStringWithMaxWeight(strings[i],weight[i],head)
# print(findSubStringWithMaxWeight(head,"ben"))
# ////////////////////////////////////////////////////////////////////////////////

def printAllSubstr(head,output):
    if len(head.children)==0:
        print(output)

    for child in head.children:
        printAllSubstr(head.children.get(child),output+child)

def helpmepradumana(head,substr):
    curr=head
    for char in substr:
        child=curr.children.get(char,None)
        if child:
            curr=child
        else:
            print(-1)
            return -1
    printAllSubstr(curr,substr)

# strings=["dog","dont","doll","deaf"]
# head=StringTrieNode()
# for string in strings:
#     insertForFindSubStringWithMaxWeight(string,0,head)
# helpmepradumana(head,"ba")
#/////////////////////////////////////////////////////////////////////
