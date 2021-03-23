import sys
from collections import defaultdict,deque
import heapq

graph = defaultdict(list)
N,M=map(int,input().split())

for _ in range(M) :
    u,v = map(int,input().split())
    graph[u].append(v)
    graph[v].append(u)

q=int(input())
for i in range(q+1) :
    s,d= input().split()
    if s in graph[d] :
        print(s)
        print("YES")
    else :
        print("NO")

class Graph():
    def __init__(self,vertices=None):
        self.v=vertices
        self.graph=defaultdict(list)

    def addEdge(self,u,v):
        self.graph[u].append(v)
        # self.graph[v].append(u)

def bsf(graph,start):
    q=deque()
    visit=defaultdict(int)
    visit[start]=1
    q.append(start)
    while q:
        curr=q.popleft()
        print(curr,end=" ")
        for item in graph[curr]:
            if visit.get(item) !=1:
                # print(item,end=" ")
                visit[item]=1
                q.append(item)

def dfs(graph,start,visited):

    if visited.get(start)!=1:
        print(start,end=" ")
        visited[start]=1
        for item in graph[start]:
            dfs(graph,item,visited)



def dfsStarter(graph,start):
    if len(graph) is 0:
        return
    else:
        visited=defaultdict(int)
        dfs(graph,start,visited)

def printShortestPath(graph,start,dest):
    q=deque()
    visited=defaultdict(int)
    q.append(start)
    parent=defaultdict(int)
    visited[start]=0
    while q:
        curr=q.popleft()
        for item in graph[curr]:
            if visited.get(item) is None:
                visited[item]=visited[curr]+1
                parent[item]=curr
                q.append(item)
            if visited.get(dest) != None:
                temp=dest
                while temp and temp!= start:
                    print(temp," <-",end=" ")
                    temp=parent[temp]
                print(start)
                return

# Greedy algorithm for finding the minimum single source distance of all nodes
def disjkstraAlgo(arr,start):
    # graph has(sourceNode,destinationNode,pathWeight)
    graph=defaultdict(list)
    for item in arr:
        src,dest,weight=item
        graph[src].append((weight,dest))

    # seen will keep record of visited nodes
    # dist will keep the current min. distance
    # pq(priority queue) will keep the node that are remains to explore as (minDistOfNode,Node), node can have multiple entries inside pq
    seen=defaultdict(bool)
    dist=defaultdict(int)
    seen[start]=True
    dist[start]=0
    pq=[(0,start)]
    heapq.heapify(pq)
    while pq:
        nodeDist,node=heapq.heappop(pq)
        for childDist,child in graph[node]:
            if seen.get(child) is None:
                seen[child]=True
                dist[child]=childDist+nodeDist
                heapq.heappush(pq,(childDist+nodeDist,child))
            elif nodeDist+childDist<dist[child] :
                dist[child]=nodeDist+childDist
                heapq.heappush(pq,(nodeDist+childDist,child))
    print(dist)

def topologicalSortByDfs(graph,start,visited,ans):
    if not visited[start]:
        visited[start]=True
        for node in graph.get(start,[]):
            topologicalSortByDfs(graph,node,visited,ans)
        ans.append(start)

def topologicalSortByDfsInitiater(graph):
    visited=defaultdict(bool)
    ans=deque()
    for key in graph:
        visited[key]=False
    for node in graph:
        if not visited[node]:
            topologicalSortByDfs(graph,node,visited,ans)
    while len(ans)>1:
        print(ans.pop(),"->",end=" ")
    print(ans.pop())

def topologicalSortByBfs(graph):
    visited=defaultdict(bool)
    indegree=defaultdict(int)
    queue=deque()
    for node in graph:
        if not indegree.get(node):
            indegree[node]=0
        for child in graph[node]:
            indegree[child]+=1
    for key in indegree:
         if indegree[key]==0:
            queue.append(key)
    while queue:
        curr=queue.popleft()
        print(curr,end="->")
        for child in graph[curr]:
            indegree[child]-=1
            if indegree[child]==0:
                queue.append(child)


def cycleDetectionUsingBfs(graph,start,visited,parent):
    q=deque()
    q.append(start)
    visited[start]=True
    parent[start]=-1
    while q:
        curr=q.popleft()
        for child in graph.get(curr,[]):
            if visited.get(child):
                if parent[child]!=curr:
                    for item in graph.get(child,[]):
                        if visited[item]:
                            return True
            else:
                visited[child]=True
                parent[child]=curr
                q.append(child)
    return False

def cycleDetectionUsingBfsInitiater(graph):
    visited=defaultdict(bool)
    parent=defaultdict(int)
    for i in graph:
        if not visited[i]:
            if not cycleDetectionUsingBfs(graph,i,visited,parent):
                continue
            else:
                return True
        return False

def cycleDetectionUsingDfs(graph,start,visited,inStack):
    if visited.get(start) and inStack.get(start):
        return True
    visited[start]=True
    inStack[start]=True
    for child in graph.get(start,[]):
        if cycleDetectionUsingDfs(graph,child,visited,inStack):
            return True
    inStack[start]=False
    return False
def cycleDetectionUsingDfsInitiater(graph):
    visited=defaultdict(bool)
    inStack=defaultdict(bool)
    for child in graph:
        if not visited[child]:
            if cycleDetectionUsingDfs(graph,child,visited,inStack):
                print(visited,inStack)
                return True
    print(visited,inStack)
    return False


# Kruskal's Algorithm

def findParent(parent, src):
    if parent[src]<0:
        return src
    return findParent(parent,parent[src])


def isDiffrentParent(src, dest,parent):

    # Union by ranking and path compression algorithm used(source: Abdul Barik (Disjoint set))
    srcParent=findParent(parent,src)
    destParent=findParent(parent,dest)

    # page compression
    if src!=srcParent:
        parent[src]=srcParent
    if destParent!=dest:
        parent[dest]=destParent
    # print(srcParent,destParent)

    # Union by ranking
    if srcParent!=destParent:
        # print(srcParent,destParent)
        if parent[srcParent]>parent[destParent]:
            parent[destParent]=parent[destParent]+parent[srcParent]
            parent[srcParent]=destParent
        else:
            parent[srcParent]=parent[destParent]+parent[srcParent]
            parent[destParent]=srcParent
        return True
    return False


def kruskalsAlgorithmForMst(edges):
    output=[]
    edges=sorted(edges,key=lambda item:item[2])
    # print(edges)
    parent=defaultdict(int)
    for edge in edges:
        parent[edge[0]]=-1
        parent[edge[1]]=-1
    for edge in edges:
        if isDiffrentParent(edge[0],edge[1],parent):
            output.append(edge)
    print(parent)
    return output

# /////////////////////////////////////////////////////////////////////////////////

def PrimsAlgo(graph,start):
    visited=defaultdict(bool)
    parents=defaultdict(int)
    parents[start]=-1
    weights=defaultdict(int)
    # node=(weight,src)
    nodes=[(0,start)]
    heapq.heapify(nodes)
    while nodes:
        src=heapq.heappop(nodes)[1]
        # print(src)
        if not visited.get(src):
            for child in graph.get(src):
                # print(child)
                if (not visited.get(child[1])) and child[0]<weights.get(child[1],sys.maxsize):
                    weights[child[1]]=child[0]
                    parents[child[1]]=src
                    heapq.heappush(nodes,child)
            visited[src]=True
    for key in graph.keys():
        print((parents[key],key,weights[key]))

def PrimsAlgoInitiater(arr,start):
    graph=defaultdict(list)
    for node in arr:
        src,dest,weight=node
        graph[src].append((weight,dest))
        graph[dest].append((weight,src))
    print(graph)
    return PrimsAlgo(graph,start)

def journeyToMoon( graph,visited,start):
    visited[start]=True
    count=0
    for child in graph.get(start,[]):
        if not visited.get(child,None):
            count+=journeyToMoon(graph,visited,child)
    count+=1
    return count

def journeyToMoonHelper(n,graph):
    visited=defaultdict(bool)
    ans=n*(n-1)//2
    for node in graph:
        if not visited.get(node,None):
            result=journeyToMoon( graph,visited,node)
            ans-=(result*(result-1))//2
    return ans

# CB-FAANG Kill Process
def kill_process(process,parent,kill):
    print("hii")
    graph=defaultdict(list)
    for i in range(len(process)):
        graph[parent[i]].append(process[i])
    queue=deque()
    queue.append(kill)
    sorted_list=[]
    print(graph)
    # while queue:
    #     vertex=queue.popleft()
    #     sorted_list.append(vertex)
    #     for child in graph.get(vertex,[]):
    #         queue.append(child)
    print(sorted_list)

process=[1,3,10,5]
parent=[3,0,5,3]
kill=5
kill_process(process,parent,kill)







