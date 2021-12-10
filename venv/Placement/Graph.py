from collections import defaultdict,deque
import heapq
from math import inf

def jumpToZero(arr, pos, visited):
    if pos >= len(arr) or pos < 0 or pos in visited:
        return False

    if arr[pos] == 0:
        return True
    visited.add(pos)
    return jumpToZero(arr, pos - arr[pos], visited) or jumpToZero(arr, pos + arr[pos], visited)

# arr = [1,2,1,1,0,1]
# pos =2
# visited = set()
# print(jumpToZero(arr, pos, visited))

def bfsShotestPath(graph,node,parent):
    distance = defaultdict(lambda :inf)
    queue = deque()
    queue.append(node)
    distance[node] = 0
    parent[node] = -1
    while queue :
        curr_node = queue.popleft()
        for child in graph[curr_node]:
            if distance[child] == inf:
                distance[child] = distance[curr_node] +1
                queue.append(child)
                parent[child] = curr_node


def wordLadder(words, src, dest):
    graph = defaultdict(list)
    if src not in words:
        words.append(src)
    for i in range(len(words)):
        for j in range(len(words)):
            if len(words[i]) == len(words[j]):
                count = 0
                for k in range(len(words[i])):
                    if words[i][k] != words[j][k]:
                        count+=1
                    if k == len(words[i])-1 and count==1:
                        graph[words[i]].append(words[j])
    parent = defaultdict(int)
    bfsShotestPath(graph,src,parent)
    node = dest
    while parent[node] != -1:
        print(node, end= "<-")
        node = parent[node]
    print(node)

# words = ["hot","dog","dot","lot","log","cog"]
# src = "hit"
# dest = "cog"
# wordLadder(words, src, dest)

def isValid(x,y,n):
    if x>=0 and x<n and y>=0 and y<n:
        return True
    return False

def knightStep(n,x1,y1,x2,y2):
    grid = [[-1 for i in range(n)] for j in range(n)]
    grid[x1][y1] = 0
    queue = deque()
    queue.append((x1,y1))
    points = [(-2,-1), (-2, 1), (-1,-2), (-1, 2), (1,-2), (1,2), (2,-1), (2,1)]
    while queue:
        x,y = queue.popleft()
        for dx,dy in points:
            if  isValid(x+dx,y+dy,n) and grid[x+dx][y+dy]==-1 :
                queue.append((x+dx, y+dy))
                grid[x+dx][y+dy]=grid[x][y] +1
        if grid[x2][y2] != -1:
            return grid[x2][y2]

# n=6
# x1= 4-1
# y1= 5-1
# x2= 1-1
# y2= 1-1
# print(knightStep(n,x1,y1,x2,y2))

def dfsTopologicalSort(node,visited,list,graph):
    visited.add(node)
    for child in graph[node]:
        if child not in visited:
            dfsTopologicalSort(child, visited,list,graph)
    list.append(node)

def courseSheduling(n, arr):
    graph = defaultdict(list)
    for edge in arr:
        graph[edge[1]].append(edge[0])
        if edge[0] not in graph:
            graph[edge[0]] = []
    visited = set()
    ans = []
    for node in graph:
        if node not in visited:
            dfsTopologicalSort(node,visited,ans,graph)
    final_ans =[x for x in reversed(ans)]
    return final_ans

# arr= [[1,0],[2,0],[3,1],[3,2]]
# print(courseSheduling(len(arr), arr))

def dijkstra(arr,src):
    graph = defaultdict(list)
    for u,v,w in arr:
        graph[u].append((v,w))
        graph[v].append((u,w))

    distance = defaultdict(lambda :inf)
    distance[src] = 0
    pq = [(0,src)]

    while pq:
        curr_dist, node = heapq.heappop(pq)
        for child, edge_wt in graph[node]:
            if distance[child] > distance[node] + edge_wt:
                distance[child] = distance[node] + edge_wt
                heapq.heappush(pq,(distance[child], child))
    print(distance)

# arr = [[0, 1, 4],
#        [0, 7, 8],
#        [1, 2, 8],
#        [1, 7, 11],
#        [2, 3, 7],
#        [2, 8, 2],
#        [2, 5, 4],
#        [3, 4, 9],
#        [3, 5, 14],
#        [4, 5, 10],
#        [5, 6, 2],
#        [6, 7, 1],
#        [6, 8, 6],
#        [7, 8, 7]]
# dijkstra(arr, 0)

# Find the number of islands https://practice.geeksforgeeks.org/problems/find-the-number-of-islands/1#

def isValidCountIsland(row,col,grid):
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 1:
        return True
    return False

def dfsCountIsland(grid,row, col):
    grid[row][col] = 2
    point = [-1,0,1]
    for dx in point:
        for dy in point:
            if isValidCountIsland(row+dx,col+dy,grid):
                dfsCountIsland(grid,row+dx,col+dy)

def countIsland(grid):
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]==1:
                count+=1
                dfsCountIsland(grid,i,j)
    return count

# grid = [[0,1],[1,0],[1,1],[1,0]]
# print(countIsland(grid))

# https://leetcode.com/problems/path-with-minimum-effort/
def isValidPoint(x,y,grid):
    if 0<= x < len(grid) and 0<=y<len(grid[0]):
        return True
    return False


def pathWithMinEffort(grid):
    points = [(0,-1),(-1,0),(0,1),(1,0)]
    distance = [[inf]*len(grid[0]) for i in range(len(grid))]
    pq = [(0,0,0)]
    distance[0][0] = 0
    while pq :
        diff,x,y = heapq.heappop(pq)
        if x== len(grid)-1 and y== len(grid[0])-1:
            print(pq)
            for i in range(len(distance)):
                print(distance[i])
            return diff
        for dx,dy in points:
            if isValidPoint(x+dx,y+dy,grid):
                new_diff = max(distance[x][y], abs(grid[x][y]-grid[x+dx][y+dy]))
                if distance[x+dx][y+dy] > new_diff:
                    # print(distance[x+dx][y+dy],new_diff, x + dx, y + dy)
                    distance[x+dx][y+dy] = new_diff
                    heapq.heappush(pq,(new_diff,x+dx,y+dy))

# grid = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
# print(pathWithMinEffort(grid))

# 1514. Path with Maximum Probability
def maxProbability(edges, succProb, start, end):
    graph = defaultdict(list)
    for i,edge in enumerate(edges):
        graph[edge[0]].append((-succProb[i],edge[1]))
        graph[edge[1]].append((-succProb[i],edge[0]))

    prob = defaultdict(lambda : 0)
    prob[start] = -1
    pq = [(-1,start)]
    while pq:
        curr_prob, node = heapq.heappop(pq)
        for child_prob, child in graph[node]:
            if prob[child]> -(curr_prob*child_prob):
                prob[child]= -(curr_prob*child_prob)
                heapq.heappush(pq,(prob[child],child))
    print(prob)
    return prob[end]

# edges = [[0,1],[1,2],[0,2]]
# succProb = [0.5,0.5,0.2]
# start = 0
# end = 2
# print(maxProbability(edges, succProb, start, end))

# Using Path compression and Union by Rank in DSU
# edgeList -> [(u,v,w),....]
def find(x, parent):
    if parent[x]==-1:
        return x
    parent[x] = find(parent[x],parent)
    return parent[x]

def union(u,v,parent, rank):
    s1 = find(u, parent)
    s2 = find(v, parent)

    if s1!=s2:
        if rank[s1]> rank[s2]:
            parent[s2] = s1
            rank[s1]+= rank[s2]
        else:
            parent[s1] = s2
            rank[s2]+=rank[s1]

def kruskelAlgoMst(edgeList,n):
    newEdgeList = []
    for u,v,w in edgeList:
        newEdgeList.append((w,u,v))

    parent = [-1]*n
    rank = [1]*n
    newEdgeList = sorted(newEdgeList)
    ans = 0
    for w, u, v in newEdgeList:
        if find(u,parent) != find(v, parent):
            union(u,v, parent, rank)
            ans+=w
    # print(parent)
    # print(rank)
    return ans

# arr = [[0,1,1],
#        [1,3,3],
#        [3,2,4],
#        [2,0,2],
#        [0,3,2],
#        [1,2,2]]
# print(kruskelAlgoMst(arr,10))

# Prim's algorithm
def primsAlgo(edgeList, n):
    graph = defaultdict(list)
    for u,v,w in edgeList:
        graph[u].append((w,v))
        graph[v].append((w,u))
    visited = set()
    pq = [(0,0)]
    print(graph)
    cost = 0
    while pq:
        curr_wt, curr_node = heapq.heappop(pq)
        if curr_node not in visited:
            cost+=curr_wt
            visited.add(curr_node)
            for child_wt, child in graph[curr_node]:
                heapq.heappush(pq,(child_wt, child))
    return cost

# arr = [[0,1,1],
#        [1,3,3],
#        [3,2,4],
#        [2,0,2],
#        [0,3,2],
#        [1,2,2]]
# print(primsAlgo(arr,10))

def floydWarshall(grid):

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == -1:
                grid[i][j] = inf

    for k in range(len(grid)):
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] > grid[row][k] + grid[k][col]:
                    grid[row][col] = grid[row][k] + grid[k][col]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == inf:
                grid[i][j] = -1
    return grid

# grid = [[0,1,43],[1,0,6],[-1,-1,0]]
# print(floydWarshall(grid))

# 997. Find the Town Judge
def findTownJudge(edges, n):
    freq = defaultdict(int)
    isPresent = set()
    for u,v in edges:
        freq[v]+=1
        isPresent.add(u)

    count = 0
    for node in freq:
        if freq[node] == n-1 and node not in isPresent:
            return node
    return -1

# n = 2
# trust = [[1,2]]
# print(findTownJudge(trust, n))

# 1557. Minimum Number of Vertices to Reach All Nodes
def findSmallestSetOfVertices( n: int, edges):
    graph = defaultdict(list)
    indegree = defaultdict(int)
    ans = []
    for u,v, in edges:
        graph[u].append(v)
        indegree[v]+=1
        indegree[u] = indegree[u]
    for node, value in indegree.items():
        if value == 0:
            ans.append(node)
    return ans

# n= 6
# edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]
# print(findSmallestSetOfVertices(n, edges))


def isSafe(graph, m, node, v, color, i):

    for child in graph[node]:
        if graph[node][child]==1:
            if color[child] == i:
                return False
    return True


def MColoring(graph, m, node, v,color):
    if node == v:
        return True

    for i in range(m):
        if isSafe(graph, m, node, v, color, i):
            color[node] = i
            if MColoring(graph, m, node+1, v, color):
                print(node+1,color)
                return True
    color[node] = -1
    return False

def MColoringHelper(graph, m, V):

    color = defaultdict(lambda : -1)
    ans = MColoring(graph,m,0,V,color)
    print(color)
    return ans

# edges = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
# m = 3
# v = 4
# print(MColoringHelper(edges, m, v))

#959. Regions Cut By Slashes
def regionsBySlashes(grid):
    parent = defaultdict(lambda : -1)
    n = len(grid)

    def find(x):
        if parent[x] == -1:
            return x
        parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        s1 = find(x)
        s2 = find(y)
        if s1 != s2:
            parent[s1] = s2

    for i in range(n):
        for j in range(n):
            if i :
                union((i - 1, j, 2), (i, j, 0))

            if j:
                union((i, j - 1, 1), (i, j, 3))

            if grid[i][j] != "/":
                union((i,j,0),(i,j,1))
                union((i,j,2),(i,j,3))
                
            if grid[i][j] != "\\":
                union((i, j, 0), (i, j, 3))
                union((i, j, 2), (i, j, 1))
    return len(set(map(find, parent)))

# grid = ["/\\","\\/"]
# print(regionsBySlashes(grid))

# 1129. Shortest Path with Alternating Colors
#  Red -> 0
#  Blue -> 1
def shortestAlternatingPaths(n, red_edges, blue_edges):
    graph = defaultdict(list)
    distance = [inf]*n
    visited = set()

    for u,v in red_edges:
        graph[u].append((v,0))

    for u,v in blue_edges:
        graph[u].append((v,1))

    queue = deque()
    queue.append((0, -1, 0))
    visited.add((0, -1, -1))
    distance[0] = 0
    while queue:
        node, node_color, node_distance = queue.popleft()
        visited.add((node,node_color))
        for child, child_color in graph[node]:
            if child_color != node_color and (child, child_color) not in visited:
                queue.append((child,child_color,node_distance+1))
                if distance[child] > node_distance+1:
                    distance[child] = node_distance + 1
    return distance

# n = 5
# red_edges = [[0,1],[1,2],[2,3],[3,4]]
# blue_edges = [[1,2],[2,3],[3,1]]
# print(shortestAlternatingPaths(n, red_edges, blue_edges))

#841. Keys and Rooms
def isRoomsOpen(rooms):
    visited = set()
    def dfs(node):
        visited.add(node)
        for child in rooms[node]:
            if child not in visited:
                dfs(child)
    dfs(0)
    print(visited)
    return len(rooms)==len(visited)

# rooms = [[1],[2],[3],[]]
# print(isRoomsOpen(rooms))

# 1466. Reorder Routes to Make All Paths Lead to the City Zero
def minReorder( n, connections):
    graph = defaultdict(list)
    parent = defaultdict(list)
    visited = set()
    count = [0]
    for u,v in connections:
        graph[u].append(v)
        parent[v].append(u)
    print(graph)
    print(parent)
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for child in graph[node]:
            dfs(child)
        for child in parent[node]:
            if child not in visited:
                count[0] += 1
                dfs(child)
    dfs(0)
    return n - count[0] -1


# n = 3
# connections = [[1,0],[2,0]]
# print(minReorder(n, connections))

# 947. Most Stones Removed with Same Row or Column

def removeStones(stones):
    parent = defaultdict(lambda :-1)

    def find(node):
        parent.setdefault(node,node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(x, y):
        parent[find(x)] = find(y)

    for row, col in stones:
        union(row, ~col)
    print(parent)
    ans = len(stones) - len({find(i) for i in parent})

    return ans
# stones = [[0,0]]
# print(removeStones(stones))

# 802. Find Eventual Safe States
def eventualSafeNodes(graph):
    rgraph = defaultdict(list)
    graph = list(map(set,graph))
    ans = []
    queue = deque()
    for i, children in enumerate(graph):
        if len(graph[i]) > 0:
            for child in children:
                rgraph[child].append(i)
        else:
            queue.append(i)

    while queue:
        node = queue.popleft()
        ans.append(node)
        for parent in rgraph[node]:
            graph[parent].remove(node)
            if len(graph[parent]) == 0:
                queue.append(parent)
    return sorted(ans)

graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
print(eventualSafeNodes(graph))



def nearestExit(maze, entrance):
    visited = set()
    x, y = entrance[0], entrance[1]
    points = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neglect = set()

    def isValid(row, col):
        if 0 <= row < len(maze) and 0 <= col < len(maze[0]):
            return True
        return False

    for dx, dy in points:
        if isValid(x + dx, y + dy) == False:
            neglect.add((x + dx, y + dy))

    queue = deque()

    queue.append((x, y, 0))
    while queue:
        row, col, curr_dist = queue.popleft()
        visited.add((row, col))
        for dx, dy in points:
            if isValid(row + dx, col + dy):
                if ((row + dx, col + dy) not in visited) and maze[row + dx][col + dy] != "+":
                    queue.append((row + dx, col + dy, curr_dist + 1))
            elif (row + dx, col + dy) not in neglect:
                return curr_dist
        print(queue)
    return -1

# maze = [["+","+","+"],[".",".","."],["+","+","+"]]
# entrance = [1,0]
# print(nearestExit(maze, entrance))

# 1615. Maximal Network Rank
def maximalNetworkRank(n, roads):
    graph = defaultdict(list)
    edges = set()
    for u,v in roads:
        graph[u].append(v)
        graph[v].append(u)
        edges.add((u,v))
        edges.add((v,u))
    # print(graph)
    ans = -1
    for u in range(n):
        for v in range(n):
            if u != v:
                temp = len(graph[u])+len(graph[v])- (1 if (u,v) in edges else 0)
                # print(u,v,temp)
                ans = max(ans,temp)
    return ans

# n = 5
# roads = [[0,1],[0,3],[1,2],[1,3],[2,3],[2,4]]
# print(maximalNetworkRank(n, roads))









































