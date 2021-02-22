from collections import defaultdict
import heapq

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
                dist[child]=dist[node]+childDist
                heapq.heappush(pq,(childDist,child))
            elif dist[node]+childDist<dist[child] :
                print("Hiii",child,dist[node]+childDist,dist[child])
                dist[child]=dist[node]+childDist
                heapq.heappush(pq,(childDist,child))
    print(dist)

l = [("A", "B", 7),
     ("A", "D", 5),
     ("B", "C", 8),
     ("B", "D", 9),
     ("B", "E", 50),
     ("C", "E", 5),
     ("D", "E", 15),
     ("D", "F", 100),
     ("E", "F", 8),
     ("E", "G", 9),
     ("F", "G", 11)]

disjkstraAlgo(l,"A")

