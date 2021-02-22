import sys
C=[[0]*2 for x in range(2)]
C[1][1]=2
print(C)
# When we want that a single variable will hold the whole list of integers
def get_array(): return list(map(int, sys.stdin.readline().strip().split()))
Arr = get_array()

# When we want that a single reference variable will hold this string
def get_string(): return sys.stdin.readline().strip()
string = get_string()

# Making 2d matrix ( or any n dim matrix ) of 0s
n=5
mat = [[0 for i in range(n)] for j in range(n)]

