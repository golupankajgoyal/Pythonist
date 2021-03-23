# BFS for graph including a level dict that keep the information of level of every child in graph
from collections import deque, defaultdict

# lambas is used to give the default value when key is not found inside defaultdict
# d=defaultdict(lambda : -1)
queue = deque([1])
graph = defaultdict(list)
level = {1: 1}
visited = []

for _ in range(int(input()) - 1):
    u, v = list(map(int, input().split()))
    graph[u].append(v)
    graph[v].append(u)

while(queue):
    curr = queue.popleft()
    if curr not in visited:
        visited.append(curr)
        for child in graph[curr]:
            if child not in visited:
                level.update({child: level[curr] + 1})
            queue.append(child)
print(list(level.values()).count(int(input())))



