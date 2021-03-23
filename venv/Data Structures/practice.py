from collections import deque, defaultdict
import sys
import heapq

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


# Construct generic graph
# n = 13
# g = Graph(n)
# g.addEdge(1, 2)
# g.addEdge(1, 3)
# g.addEdge(1, 4)
# g.addEdge(2, 1)
# g.addEdge(2, 11)
# g.addEdge(3, 1)
# g.addEdge(3, 4)
# g.addEdge(3, 6)
# g.addEdge(4, 3)
# g.addEdge(4, 5)
# g.addEdge(4, 1)
# g.addEdge(5, 4)
# g.addEdge(5, 7)
# g.addEdge(6, 3)
# g.addEdge(6, 7)
# g.addEdge(6, 8)
# g.addEdge(6, 9)
# g.addEdge(7, 5)
# g.addEdge(7, 6)
# g.addEdge(8, 6)
# g.addEdge(9, 6)
# g.addEdge(9, 10)
# g.addEdge(10, 11)
# g.addEdge(10, 9)
# g.addEdge(11, 12)
# g.addEdge(11, 13)
# g.addEdge(11, 2)
# g.addEdge(11, 10)
# g.addEdge(12, 11)
# g.addEdge(13, 11)
# dfsStarter(g.graph, 1)

# Graph for topological sort
# g = Graph(6)
# g.addEdge(5, 2)
# g.addEdge(5, 0)
# g.addEdge(4, 0)
# g.addEdge(4, 1)
# g.addEdge(2, 3)
# g.addEdge(3, 1)
# topologicalSortByDfsInitiater(g.graph)
# for kruskal's Algo


edges = [(1,3,1),
         (0,1,2),
         (0,3,3),
         (0,2,8),
         (4,5,5),
         (2,3,6),
         (3,5,7),
         (2,1,4),
         (2,4,9),
         (2,5,10),
         (3,4,11)]

