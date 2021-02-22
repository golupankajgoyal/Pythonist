from collections import defaultdict,deque

def hide_and_seek(graph,start_time,end_time,timer,root,visited):
    print(root)
    visited[root] = True
    timer[0] += 1
    start_time[root] = timer[0]
    for child in graph[root]:
        if len(graph[child]) > 0 and (not visited[child]):
            hide_and_seek(graph, start_time, end_time, timer, child, visited)
        else:
            visited[child] = True
            timer[0] += 1
            start_time[child] = timer[0]
            timer[0] += 1
            end_time[child] = timer[0]
    timer[0] += 1
    end_time[root] = timer[0]
    # return

def check(x,y,start_time,end_time):
    if start_time[x]<start_time[y] and end_time[x]>end_time[y]:
        return True
    else:
        return False

n=int(input())
graph=defaultdict(list)
for i in range(n-1):
    n1,n2=map(int,input().split())
    graph[n1].append(n2)
start_time=defaultdict(int)
end_time=defaultdict(int)
visited=defaultdict(bool)
print(graph)
hide_and_seek(graph,start_time,end_time,[0],1,visited)
q=int(input())

for _ in range(q):
    direction,x,y=map(int,input().split())
    # print(x,y)
    if (not check(x,y,start_time,end_time)) and (not check(y,x,start_time,end_time)):
        print("NO")
        continue
    if direction==0 and check(x,y,start_time,end_time):
        print("YES")
    elif direction==1 and check(y,x,start_time,end_time):
        print("YES")
    else:
        print("NO")

