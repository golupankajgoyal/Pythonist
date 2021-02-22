def permutation(string,i):
    if len(string)<=i+1:
        print("".join(string),end=" ")
        return
    for j in range(i,len(string)):
        # print(j)
        string[i],string[j]=string[j],string[i]
        permutation(string,i+1)
        string[i],string[j]=string[j],string[i]
permutation(list("abc"),0)

# /////////////////////////////////////////////////////////////
def isPossible(n, row, col, board):
    for i in range(1,row+1):
        # print(row,i,end="row")
        if board[row-i][col]==1:
            return False
        if col-i>=0 and board[row-i][col-i]==1:
            return False
        if col+i<n and board[row-i][col+i]==1:
            return False
    return True

def nQueenHelper(n, board, row):
    # print(row)
    if n==row:
        var = [[[print(board[i][j], end=" ") for j in range(n)],print("")] for i in range(n)]
        print("///////////////")
        return

    for k in range(n):
        if isPossible(n,row,k,board):
            board[row][k]=1
            nQueenHelper(n,board,row+1)
            board[row][k]=0
    return


def nQueen(n):
    board=[[0 for i in range(n)] for j in range(n)]
    nQueenHelper(n,board,0)

# /////////////////////////////////////////////////////////////////////////

def ratInMazeHelper(maze, solution, x, y,n):

    if x==n-1 and y==n-1:
        solution[x][y]=1
        ans=[[[print(solution[i][j],end=" ")for j in range(n)],print("")]for i in range(n)]
        print("//////////////")
        return
    if x<0 or y<0 or x>=n or y>=n or solution[x][y]==1 or maze[x][y]==0:
        return
    solution[x][y]=1
    ratInMazeHelper(maze,solution,x-1,y,n)
    ratInMazeHelper(maze,solution,x+1,y,n)
    ratInMazeHelper(maze,solution,x,y-1,n)
    ratInMazeHelper(maze,solution,x,y+1,n)
    solution[x][y]=0


def ratInMaze(maze,n):
    solution=[[0 for j in range(n)]for i in range(n)]
    ratInMazeHelper(maze,solution,0,0,n)
    return
# ///////////////////////////////////////////////////////////////////////////////
def isPossible(stalls, c, dist):
    count=0
    last=stalls[0]
    for i in stalls[1:]:
        if dist<=(i-last):
            count+=1
            last=i
        if count==c-1:
            return True
    return False


def arrangeCowsHelper(stalls, c, start, end,ans):
    if start>=end:
        if isPossible(stalls,c,start):
            return start
        else:
            return ans[0]
    dist=start + (end-start)//2
    print(dist,start,end)
    if isPossible(stalls,c,dist):
        ans[0]=dist
        start=dist+1
    else:
        end=dist-1
    return arrangeCowsHelper(stalls,c,start,end,ans)

def arrangeCows(stalls,c,n):
    stalls=sorted(stalls)
    ans=[0]
    return arrangeCowsHelper(stalls,c,1,stalls[n-1]-stalls[0],ans)
# /////////////////////////////////////////////////////////////////////////
def merging(arr, start, mid, end):
    temp=[]
    count=0
    i=start
    j=mid+1
    while i<=mid and j<=end:
        if arr[i]>arr[j]:
            count+=(mid-i+1)
            temp.append(arr[j])
            j+=1
        else:
            temp.append(arr[i])
            i+=1
    while i<= mid:
        temp.append(arr[i])
        i+=1
    while j<=end:
        temp.append(arr[j])
        j+=1
    for i in range(len(temp)):
        arr[start+i]=temp[i]
    return count


def findInversionCountHelper(arr, start, end):

    if end<=start:
        return 0
    mid=start+(end-start)//2
    leftInv=findInversionCountHelper(arr,start,mid)
    rightInv=findInversionCountHelper(arr,mid+1,end)
    mergeInv=merging(arr,start,mid,end)
    return leftInv+rightInv+mergeInv


def findInversionCount(arr):
    return findInversionCountHelper(arr,0,len(arr)-1)

# ////////////////////////////////////////////////////////////

def printSuduko(mat,size):
    val=[[[print(mat[i][j],end=" ") for j in range(size)],print("")]for i in range(size)]
    return

def isPossible(mat, row, col, i, size):
    for x in range(size):
        if mat[row][x]==i or mat[x][col]==i:
            return False
    den=int(size**(1/2))
    sx=(row//den)*den
    sy=(col//den)*den
    for j in range(sx,sx+den):
        for k in range(sy,sy+den):
            if mat[j][k]==i:
                return False
    return True

def sudukoSolverHelper(mat,row,col,size):
    if row==size:
        printSuduko(mat,size)
        return True

    if col==size:
        return sudukoSolverHelper(mat,row+1,0,size)

    if mat[row][col]!=0:
        return sudukoSolverHelper(mat,row,col+1,size)

    for i in range(1,size+1):
        if isPossible(mat,row,col,i,size):
            mat[row][col]=i
            if sudukoSolverHelper(mat,row,col+1,size):
                return True
    mat[row][col]=0
    return False



def sudukoSolver(mat,n):
    print(len(mat))
    sudukoSolverHelper(mat,0,0,n)
