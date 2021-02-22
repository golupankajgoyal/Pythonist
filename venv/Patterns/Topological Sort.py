from collections import defaultdict,deque

def topological_sort(vertices,edges):
    graph=defaultdict(list)
    for edge in edges:
        parent,child=edge[0],edge[1]
        if child not in graph:
            graph[child]=[]
        if parent in graph:
            graph[parent].append(child)
        else:
            graph[parent]=[child]
    indegree=defaultdict(int)
    for vertex in graph:
        indegree[vertex]=0
    for vertex in graph:
        child_list=graph[vertex]
        for child in child_list:
            indegree[child]+=1
    sources=deque()
    for vertex in indegree:
        if indegree[vertex]==0:
            sources.append(vertex)
    sorted_list=[]
    while sources:
        vertex=sources.popleft()
        sorted_list.append(vertex)
        for child in graph[vertex]:
            indegree[child]-=1
            if indegree[child]==0:
                sources.append(child)
    print(sorted_list)

# vertices=5
# Edges=[[4, 2], [4, 3], [2, 0], [2, 1], [3, 1]]
# topological_sort(vertices,Edges)

#There is a dictionary containing words from an alien language for which
# we don’t know the ordering of the characters. Write a method to find
# the correct order of characters in the alien language.
def find_correct_order(arr):
    graph=defaultdict(list)
    indegree=defaultdict(int)
    for i in range(1,len(arr)):
        word1,word2=arr[i-1],arr[i]
        for j in range(min(len(word1),len(word2))):
            parent,child=word1[j],word2[j]
            if parent!=child:
                graph[parent].append(child)
                indegree[child]+=1
                if parent not in indegree:
                    indegree[parent]=0
                break
    source=deque()
    for vertex in indegree:
        if indegree[vertex]==0:
            source.append(vertex)
    sorted_list=[]
    while source:
        vertex=source.popleft()
        sorted_list.append(vertex)
        for child in graph[vertex]:
            indegree[child]-=1
            if indegree[child]==0:
                source.append(child)
    print(sorted_list)

# arr=["ywx", "wz", "xww", "xz", "zyy", "zwz"]
# find_correct_order(arr)

# Given a sequence originalSeq and an array of sequences, write a method to find
# if originalSeq can be uniquely reconstructed from the array of sequences.
# Unique reconstruction means that we need to find if originalSeq is the only sequence
# such that all sequences in the array are subsequences of it.
def reconstruct_sequence(arr,org_seq):
    graph=defaultdict(list)
    indegree=defaultdict(int)
    for seq in arr:
        for i in range(len(seq)-1):
            parent,child=seq[i],seq[i+1]
            graph[parent].append(child)
            indegree[child]+=1
            if parent not in indegree:
                indegree[parent]=0
    source=deque()
    for vertex in indegree:
        if indegree[vertex]==0:
            source.append(vertex)
    sorted_list=[]
    while source:
        if len(source)>1:
            return False
        if org_seq[len(sorted_list)]!=source[0]:
            False
        num=source.popleft()
        sorted_list.append(num)
        for child in graph[num]:
            indegree[child]-=1
            if indegree[child]==0:
                source.append(child)
    # print(sorted_list)
    return True
originalSeq= [3, 1, 4, 2, 5]
seqs= [[3, 1, 5], [1, 4, 2, 5]]
print(reconstruct_sequence(seqs,originalSeq))











