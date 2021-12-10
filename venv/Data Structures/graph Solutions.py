from collections import defaultdict,deque
import heapq
from math import inf
# CB-FAANG Kill Process
def kill_process(process,parent,kill):
    graph=defaultdict(list)
    for i in range(len(process)):
        graph[parent[i]].append(process[i])
    queue=deque()
    queue.append(kill)
    sorted_list=[]
    while queue:
        vertex=queue.popleft()
        sorted_list.append(vertex)
        for child in graph.get(vertex,[]):
            queue.append(child)
    print(sorted_list)

# process=[1,3,10,5,4,8,15,20]
# parent=[3,0,5,3,5,5,8,8]
# kill=5
# kill_process(process,parent,kill)

# CB-FAANG JUMP to zeros
def jump_to_zero(arr,start):
    graph=defaultdict(list)
    for i in range(len(arr)):
        if arr[i]==0:
            graph[i]=[]
        else:
            if i+arr[i]<len(arr):
                graph[i].append(i+arr[i])
            if i-arr[i]>=0:
                graph[i].append(i-arr[i])
    visited=defaultdict(bool)
    queue=deque()
    queue.append(start)
    while queue:
        vertex=queue.popleft()
        visited[vertex]=True
        if len(graph[vertex])==0:
            return True
        else:
            for child in graph[vertex]:
                if  not visited[child]:
                    queue.append(child)
    return False

# arr=[4,2,3,0,3,1,2]
# print(jump_to_zero(arr,3))

# CB- FAANG Word ladder problem
def word_ladder(arr,start,end):
    words=set(arr)
    words.add(start)
    words.add(end)
    graph=defaultdict(list)
    for word in words:
        for char in word:
            for val in range(ord("a"),ord("z")+1):
                if chr(val)!=char:
                    new_word=word.replace(char,chr(val))
                    if new_word in words:
                        graph[word].append(new_word)
    queue=deque()
    visited=defaultdict(bool)
    queue.append((start,1))
    while queue:
        vertex,dist=queue.popleft()
        visited[vertex]=True
        for child in graph[vertex]:
            if child==end:
                return dist+1
            if not visited[child]:
                queue.append((child,dist+1))
    return 0

# arr=["hit","hot","dog","lot","log","cog","dot"]
# start="hit"
# end="cog"
# print(word_ladder(arr,start,end))

# CB-FAANG Bipartite Graph
def is_bipartitie(graph):
    color=[0]*len(graph)

    for j in range(len(graph)):
        if color[j]==0:
            color[j]=1
            queue = deque()
            queue.append(j)
            while queue:
                i=queue.popleft()
                for child in graph[i]:
                    if color[child]==0:
                        color[child]=-color[i]
                        queue.append(child)
                    elif color[child]==color[i]:
                        return False
    return True

# print(is_bipartitie([[1,2,3],[0,2],[0,1,3],[0,2]]))

# CB-FAANG Dependent course
def  dependent_course(numCourses, prerequisites):
    graph=defaultdict(list)
    indegree=defaultdict((int))
    for course in prerequisites:
        child,parent=course
        graph[parent].append(child)
        indegree[child]+=1
        if parent not in indegree:
            indegree[parent]=0
    queue=deque()
    for i in indegree:
        if indegree[i]==0:
            queue.append(i)
    sorted_list=[]
    while queue:
        vertex=queue.popleft()
        sorted_list.append(vertex)
        for child in graph[vertex]:
            indegree[child]-=1
            if indegree[child]==0:
                queue.append(child)
    print(graph,sorted_list)
    return len(sorted_list)==len(graph)

# Edges=[[1,4],[2,4],[3,1],[3,2]]
# print(dependent_course(1,[]))

# CB-FAANG Rotten Oranges
def do_rotten(r,c,queue,grid):
    # print("hii")
    count=0
    if r-1>=0 and grid[r-1][c]==1:
        grid[r - 1][c] = 2
        queue.append((r-1,c))
        count+=1
    if c-1>=0 and grid[r][c-1]==1:
        grid[r][c-1] =2
        queue.append((r,c-1))
        count+=1
    if c+1<len(grid[0]) and grid[r][c+1]==1:
        grid[r][c+1] = 2
        queue.append((r,c+1))
        count+=1
    if r+1<len(grid) and grid[r+1][c]==1:
        grid[r + 1][c] =2
        queue.append((r+1,c))
        count+=1
    return count

def orangesRotting(grid):
    row=len(grid)
    col=len(grid[0])
    queue=deque()
    for i in range(row):
        for j in range(col):
            if grid[i][j]==2:
                queue.append((i,j))
    time=0
    while queue:
        size=len(queue)
        count=0
        for i in range(size):
            r,c=queue.popleft()
            count+=do_rotten(r,c,queue,grid)
        if count!=0:
            time+=1
    for i in range(row):
        for j in range(col):
            if grid[i][j]==1:
                return -1
    return time
# grid=[[2,1,1],[0,1,1],[1,0,1]]
# print(orangesRotting(grid))

# CB-FAANG Walls And Gates
def change_dist(r,c,queue,grid):
    # print("HIii")
    if r-1>=0 and grid[r-1][c]==inf:
        grid[r-1][c]=grid[r][c]+1
        queue.append((r-1,c))
    if c-1>=0 and grid[r][c-1]==inf:
        grid[r][c-1]=grid[r][c]+1
        queue.append((r , c-1))
    if c+1<len((grid[0])) and grid[r][c+1]==inf:
        grid[r][c+1]=grid[r][c]+1
        queue.append((r , c+1))
    if r+1<len(grid) and grid[r+1][c]==inf:
        grid[r+1][c]=grid[r][c]+1
        queue.append((r + 1, c))
def find_gate(grid):
    row=len(grid)
    col=len(grid[0])
    queue=deque()
    for i in range(row):
        for j in range(col):
            if grid[i][j]==0:
                queue.append((i,j))
    while queue:
        size=len(queue)
        for i in range(size):
            r,c=queue.popleft()
            change_dist(r,c,queue,grid)
# grid = [[inf, -1, 0, inf],
#         [inf, inf, inf, -1],
#         [inf, -1, inf, -1],
#         [0, -1, inf, inf]]
# find_gate(grid)
# for i in range(len(grid)):
#     print(grid[i])

# CB-FAANG Minimum Knight Moves
def is_valid(row,col,grid):
    if 0<=row<len(grid) and 0<=col<len(grid[0]) and grid[row][col]<0:
        return True
    return False

def move_knight(r,c,queue,grid):
    if is_valid(r-2,c-1,grid):
        grid[r-2][c-1]=grid[r][c]+1
        queue.append((r-2,c-1))
    if is_valid(r-2,c+1,grid):
        grid[r-2][c+1]=grid[r][c]+1
        queue.append((r-2,c+1))
    if is_valid(r-1,c-2,grid):
        grid[r-1][c-2]=grid[r][c]+1
        queue.append((r-1,c-2))
    if is_valid(r-1,c+2,grid):
        grid[r-1][c+2]=grid[r][c]+1
        queue.append((r-1,c+2))
    if is_valid(r+1,c-2,grid):
        grid[r+1][c-2]=grid[r][c]+1
        queue.append((r+1,c-2))
    if is_valid(r+1,c+2,grid):
        grid[r+1][c+2]=grid[r][c]+1
        queue.append((r+1,c+2))
    if is_valid(r+2,c-1,grid):
        grid[r+2][c-1]=grid[r][c]+1
        queue.append((r+2,c-1))
    if is_valid(r+2,c+1,grid):
        grid[r+2][c+1]=grid[r][c]+1
        queue.append((r+2,c+1))

def minimum_knight_moves(row,col,s1,s2,e1,e2):
    grid=[[-1 for j in range(col)]for i in range(row)]
    grid[e1][e2]=-2
    grid[s1][s2]=0
    queue=deque()
    queue.append((s1,s2))
    while queue:
        r,c=queue.popleft()
        move_knight(r,c,queue,grid)
        if grid[e1][e2]!=-2:
            for i in range(row):
                print(grid[i])
            return grid[e1][e2]
    return -1
# row=6
# col=6
# print(minimum_knight_moves(row,col,1,3,5,0))

#CB-FAANG Friends Group
def dfs(vertex,graph,visited):
    visited[vertex]=True
    if vertex in graph:
        for child in graph[vertex]:
            if not visited[child]:
                dfs(child,graph,visited)
def friends_group(grid):
    graph=defaultdict(list)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]==1:
                graph[i].append(j)
    visited=defaultdict(bool)
    group=0
    for vertex in graph:
        if not visited[vertex]:
            group+=1
            dfs(vertex,graph,visited)
    return group

# grid = [[1, 1, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 1],
#         [0, 0, 1, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 1, 0],
#         [0, 1, 0, 0, 0, 1]]
# print(friends_group(grid))

# CB FAANG No. Of Islands
def count_island(grid):
    graph=defaultdict(list)
    col=len(grid[0])
    for i in range(len(grid)):
        for j in range(col):
            unique_id=i*col+j
            if grid[i][j]==1:
                graph[unique_id]=[]
                if i-1>=0 and grid[i-1][j]==1:
                    child_id=(i - 1) * col + j
                    graph[unique_id].append(child_id)
                    graph[child_id].append(unique_id)
                if j-1>=0 and grid[i][j-1]==1:
                    child_id=(i)*col+j-1
                    graph[unique_id].append(child_id)
                    graph[child_id].append(unique_id)
    visited=defaultdict(bool)
    island_count=0
    for vertex in graph:
        if not visited[vertex]:
            island_count+=1
            dfs(vertex,graph,visited)
    return island_count
# grid=[[1,1,0,0,0],
#       [1,1,0,1,0],
#       [0,0,1,0,0],
#       [0,0,0,1,1]]
# print(count_island(grid))

# CB-FAANG MAximum Area Of Island
def dfs_max_area(vertex,graph,visited):
    if not visited[vertex]:
        visited[vertex]=True
        count=1
        for child in graph[vertex]:
            count+=dfs_max_area(child,graph,visited)
        return count
    else:
        return 0

def max_area(grid):
    graph=defaultdict(list)
    col=len(grid[0])
    for i in range(len(grid)):
        for j in range(col):
            unique_id=i*col+j
            if grid[i][j]==1:
                graph[unique_id]=[]
                if i-1>=0 and grid[i-1][j]==1:
                    child_id=(i-1)*col+j
                    graph[unique_id].append(child_id)
                    graph[child_id].append(unique_id)
                if j-1>=0 and grid[i][j-1]==1:
                    child_id=(i)*col+j-1
                    graph[unique_id].append(child_id)
                    graph[child_id].append(unique_id)
    visited=defaultdict(bool)
    # print(graph)
    max_area=0
    for vertex in graph:
        if not visited[vertex]:
            temp=dfs_max_area(vertex,graph,visited)
            max_area=max(max_area,temp)
    print(max_area)
#
# grid=[[1,1,0,0,0],
#       [1,1,1,0,1],
#       [1,0,0,0,1],
#       [0,0,1,1,1],
#       [1,1,1,1,1]]
# max_area(grid)

#
# CB-FAANG Dependent Course-2
def dependent_course(arr,total_courses):
    graph=defaultdict(list)
    indegree=defaultdict(int)
    for edge in arr:
        child,parent=edge
        graph[parent].append(child)
        indegree[child]+=1
        if parent not in indegree:
            indegree[parent]=0
    queue=deque()
    for vertex in indegree:
        if indegree[vertex]==0:
            queue.append(vertex)
    sorted_list=[]
    while queue:
        vertex=queue.popleft()
        sorted_list.append(vertex)
        for child in graph[vertex]:
            indegree[child]-=1
            if indegree[child]==0:
                queue.append(child)
    if len(sorted_list)==total_courses:
        print(sorted_list)
    else:
        print(-1)
# arr=[(1,0),(2,0),(3,1),(3,2)]
# total_courses=4
# dependent_course(arr,total_courses)

# Dijkistra's Algo to find SSSP for weighted graph
def find_sorted_path(edges,src):
    graph=defaultdict(list)
    for edge in edges:
        start,dest,weight=edge
        graph[start].append((dest,weight))
        graph[dest].append((start,weight))
    visited=defaultdict(bool)
    distance=defaultdict(int)
    min_heap=[]
    min_heap.append((0,src))
    visited[src]=True
    distance[src]=0
    while min_heap:
        weight,vertex=heapq.heappop(min_heap)
        for child,child_wght in graph[vertex]:
            if not visited[child]:
                visited[child]=True
                distance[child]=distance[vertex]+child_wght
                heapq.heappush(min_heap,(child_wght,child))
            elif distance[vertex]+child_wght<distance[child]:
                distance[child]=distance[vertex]+child_wght
                heapq.heappush(min_heap,(child_wght,child))
    for dist in sorted(distance.values()):
        print(dist,end=" ")

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
find_sorted_path(l,"A")

# CB-Religious People
# All the people living in our imaginary world Bitworld are very religious.
# There are N cities in Bitworld numbered from 1 to N.
# Due to a storm, every road in Bitworld was destroyed and now no temples are left.
# There are 2 types of operations :
#  1. You can construct a temple by giving away A dollars in ith city.
#  2. You can repair an already existing road by giving away B dollars.
# Your goal is to make temples such that people of every city have access to some temple.

def dfs_religious_people(vertex,graph,visited):
    count=1
    visited[vertex]=True
    for child in graph[vertex]:
            if not visited[child]:
                count+=dfs_religious_people(child,graph,visited)
    return count

def religious_prople(roads,cities_count,road_count,temple_cost,road_cost):
    graph=defaultdict(list)
    for road in roads:
        src,dest=road
        graph[src].append(dest)
        graph[dest].append(src)
    if temple_cost<road_cost:
        print(len(graph)*temple_cost)
    else:
        visited=defaultdict(list)
        cost=0
        for vertex in graph:
            if not visited[vertex]:
                cost+=temple_cost
                node_count=dfs_religious_people(vertex,graph,visited)
                print(node_count)
                cost+=((node_count-1)*road_cost)
        print(cost)
# roads = [[1, 3],
#          [3, 4],
#          [2, 4],
#          [1, 2],
#          [2, 3],
#          [5, 6]]
# cities_count=6
# road_count=6
# temple_cost=5
# road_cost=2
# religious_prople(roads,cities_count,road_count,temple_cost,road_cost)

# Given a dictionary of words, find minimum number of trials to reach from source word to destination word.
# A valid trial on word 'w' is defined as either insert, delete or substitute operation of a single character
# in word 'w' which results in a word 'w1' which is also present in the given dictionary.
# For example, for dictionary {"BCCI","AICC","ICC","CCI","MCC","MCA", "ACC"},
# minimum number of trials to reach from word "AICC" to "ICC" is 1.
# Only 1 opeartion of deleting character 'A' is required to reach from word "AICC" to word "ICC".
# Minimum number of trials to reach from "AICC" to "MCC" is 2(AICC->ICC->MCC) and
# minimum number of trials to reach from "AICC" to "MCA" is 3(AICC->ICC->MCC->MCA).

# It will count the no. of editing steps for matching the string
def edit_strings(str1,str2,dp):
    if len(str1)==0 or len(str2)==0:
        dp[len(str1)][len(str2)]=max(len(str1),len(str2))
        return dp[len(str1)][len(str2)]
    if dp[len(str1)][len(str2)]!=-1:
        return dp[len(str1)][len(str2)]
    if str1[0]==str2[0]:
        dp[len(str1)][len(str2)]=edit_strings(str1[1:],str2[1:],dp)
        return dp[len(str1)][len(str2)]
    insert=edit_strings(str1[1:],str2,dp)+1
    delete=edit_strings(str1,str2[1:],dp)+1
    replace=edit_strings(str1[1:],str2[1:],dp)+1
    dp[len(str1)][len(str2)]=min(insert,delete,replace)
    return dp[len(str1)][len(str2)]

# Create graph of given dictonary
def create_graph(words):
    graph=defaultdict(list)
    for i in range(len(words)):
        word1=words[i]
        for j in range(i+1,len(words)):
            word2=words[j]
            if abs(len(word1)-len(word2))<=1:
                dp = [[-1 for j in range(len(word2) + 1)] for i in range(len(word1) + 1)]
                if edit_strings(word1,word2, dp)==1:
                    graph[word1].append(word2)
                    graph[word2].append(word1)
    return graph

# find min. step to reach from one to word to the other in graph
def bfs(graph,word1,word2):
    queue=deque()
    queue.append((word1,0))
    visited=set()

    while queue:
        word,dist=queue.popleft()
        visited.add(word)
        for child in graph[word]:
            if child==word2:
                return dist+1
            if child not in visited:
                queue.append((child,dist+1))
    return -1


# words=["BCCI","AICC","ICC","CCI","MCC","MCA", "ACC"]
# word1="AICC"
# word2="BCCI"
# graph=create_graph(words)
# print(bfs(graph,word1,word2))


# Find a Mother Vertex in a Graph
# A mother vertex in a graph G = (V,E) is a vertex v such that
# all other vertices in G can be reached by a path from v.
def dfs_mother_vertex(vertex,visited,graph):
    visited.add(vertex)
    for child in graph[vertex]:
        if child not in visited:
            dfs_mother_vertex(child, visited, graph)

def mother_vertex(arr):
    graph=defaultdict(list)
    for edge in arr:
        n1,n2=edge[0],edge[1]
        graph[n1].append(n2)
        if n2 not in graph:
            graph[n2]=[]
    visited=set()
    mother_node=-1
    for vertex in graph:
        if vertex not in visited:
            dfs_mother_vertex(vertex,visited,graph)
            mother_node=vertex
    visited=set()
    if mother_node!=-1:
        dfs_mother_vertex(mother_node,visited,graph)
    if any(i not in visited for i in graph):
        return -1
    else:
        return mother_node

# arr = [[0, 1],
#        [0, 2],
#        [1, 3],
#        [4, 1],
#        [6, 4],
#        [5, 6],
#        [5, 2],
#        [6, 0]]
# print(mother_vertex(arr))

# Count the total number of ways or paths that exist between two vertices in a directed graph.
# These paths don’t contain a cycle, the simple enough reason is that a cycle contains
# an infinite number of paths and hence they create a problem.
def dfs_total_path(src,dst,dp,graph):
    if src==dst:
        return 1
    if dp[src]!=-1:
        return dp[src]
    ans=0
    for child in graph[src]:
        dp[child]=dfs_total_path(child,dst,dp,graph)
        ans+=dp[child]
    if ans!=0:
        dp[src]=ans
    return ans


def count_total_path(arr,src,dst):
    graph=defaultdict(list)
    for edge in arr:
        graph[edge[0]].append(edge[1])
    dp=defaultdict(lambda:-1)
    return dfs_total_path(src,dst,dp,graph)

# arr = [[1,4],[1,2],[2,3],[5,4],[4,3],[1,3],[2,5]]
# s = 1
# d = 4
# print(count_total_path(arr,s,d))


# Given a sorted dictionary of an alien language having N words and k
# starting alphabets of standard dictionary. Find the order of characters in the alien language.
#User function Template for python3
def findOrder(arr ):
    graph=defaultdict(list)
    indegree=defaultdict(int)
    for i in range(len(arr)-1):
        word1,word2=arr[i],arr[i+1]
        for j in range(min(len(word1),len(word2))):
            if word1[j]!=word2[j]:
                graph[word1[j]].append(word2[j])
                break
    for vertex in graph:
        for child in graph[vertex]:
            indegree[child]+=1
    queue=deque()
    for vertex in graph:
        if indegree[vertex]==0:
            queue.append(vertex)
    word=""
    while queue:
        char=queue.popleft()
        word+=char
        for child in graph[char]:
            indegree[child]-=1
            if indegree[child]==0:
                queue.append(child)
    # print(word)
    return word
# arr=["baa","abcd","abca","cab","cad"]
# print(findOrder(arr))

# Jumping Numbers
# Given a positive number X. Find the largest Jumping Number smaller than or equal to X.
# Jumping Number: A number is called Jumping Number if all adjacent digits in it differ by only 1.
# All single digit numbers are considered as Jumping Numbers.
# For example 7, 8987 and 4343456 are Jumping numbers but 796 and 89098 are not.
def jumpingNums( X):
    # code here
    queue = deque()
    for i in range(0, 10):
        queue.append(i)

    max_num = -1
    while queue:
        num = queue.popleft()
        if num <= X:
            max_num = max(max_num, num)
            if num % 10 == 0:
                queue.append(num * 10 + 1)
            elif num % 10 == 9:
                queue.append(num * 10 + 8)
            else:
                queue.append(num * 10 + num % 10 + 1)
                queue.append(num * 10 + num % 10 - 1)
    return max_num

# Test case
# X=50
# print(jumpingNums(X))

# Find Eventual Safe States
# https://leetcode.com/problems/find-eventual-safe-states/
def dfs(i, color, graph):
    color[i] = 1
    for child in graph[i]:
        if color[child] == 2:
            return True
        elif color[child] == 1 or dfs(child, color, graph) == False:
            return False
    color[i] = 2
    return True

def eventualSafeNodes(graph):
    ans = []
    color = defaultdict(int)
    for i in range(len(graph)):
        if color[i] == 0 and dfs(i, color, graph):
            ans.append(i)
        elif color[i] == 2:
            ans.append(i)
    return sorted(ans)

# Input:
# graph = [[1,2],[2,3],[5],[0],[5],[],[]]
# print(eventualSafeNodes(graph))

# Detect cycle in an undirected graph
def isCyclic(V, adj):
    parent = defaultdict(int)
    visited = defaultdict(bool)
    queue = deque()
    queue.append(0)
    while queue:
        node = queue.popleft()
        for child in adj[node]:
            if visited[child] == True and parent[node] != child :
                return True
            queue.append(child)
            parent[child] = node
        visited[node] = True
    return False



