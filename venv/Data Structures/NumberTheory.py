def GCD(a,b):
    if b>a:
        return GCD(b,a)
    if b==0:
        return a
    return GCD(b,a%b)

def extendedEuclid(a,b):
    if b>a:
        return extendedEuclid(b,a)
    if b==0:
        return 1,0,a
    x1,y1,gcd1=extendedEuclid(b,a%b)
    x=y1
    y=x1-y1*(a//b)
    gcd=gcd1
    return x,y,gcd

def findPrimeBYSieveOfErathosthenes(prime, n):
    prime[0]=False
    prime[1]=False
    for i in range(2,int(n**(0.5))+1):
        if prime[i]:
            index=i
            for j in range(i*i,n+1,i):
                prime[j]=False

def totalFactorsOfFactorial(n):
    ans=1
    prime=[True]*(n+1)
    findPrimeBYSieveOfErathosthenes(prime,n)
    print(prime)
    for i in range(2,n+1):
        if prime[i]:
            power=0
            factor=n//i
            while factor>=1:
                power+=factor
                factor//=i
            print(i,"=",power)
            ans*=(power+1)
    return ans


def multiply(a,b):
    r1=r2=2
    c1=c2=2
    ans=[[0 for j in range(c2)] for i in range(r1)]
    for i in range(r1):
        for j in range(c2):
            sum=0
            for k in range(r2):
                sum+=(a[i][k]*b[k][j])
            ans[i][j]=sum
    print(ans)
    return ans


def power(A,n,M):
    if n<=1:
        return A
    A=power(A,n//2,M)
    A=multiply(A,A)
    if n%2!=0:
        A=multiply(A,M)
    return A

def fibByMatrixExpo(n):
    M=[[1,1],[1,0]]
    M=power(M,n-1,M)
    return M[0][0]


def multiplyMat(a, a1):
    ans=[[0 for j in range(2)] for i in range(2)]
    for i in range(2):
        for j in range(2):
            sum=0
            for k in range(2):
                sum+=(a[i][k]*a1[k][j])
            ans[i][j]=sum
    print(ans)
    return ans

def powerMat(m, pow):
    if pow<=1:
        return m
    a=powerMat(m,pow//2)
    a=multiplyMat(a,a)
    if pow%2!=0:
        a=multiplyMat(a,m)
    return a


def calcFib(n):
    m=[[1,1],[1,0]]
    a=powerMat(m,n-1)
    return a[0][0],a[0][1]

def calcPow(num, pow, mod):
    if pow==0:
        return 1
    if pow==1:
        return num
    a=calcPow(num,pow//2,mod)
    ans=(a*a)%mod
    if pow%2!=0:
        ans=(num*ans)%mod
    return ans


def incomeOfNthDay(n,a,b,m):
    fibA,fibB=calcFib(n)
    remA=fibA%m
    remB=fibB%m
    x=calcPow(a,remA,m)
    y=calcPow(b,remB,m)
    ans=(x*y)%m
    return ans

def eulerPhi(n):
    ans=[i for i in range(n+1)]
    for i in range(2,n+1):
        if ans[i]==i:
            for j in range(i,n+1,i):
                ans[j]=(ans[j]*(i-1))//i
            ans[i]=i-1
    print(ans)

def findCubeFreeNumbers(n):
    ans=[True for i in range(n+1)]
    for i in range(2,int((n**(1/3))+1)):
        if ans[i]==True:
            factor=i*i*i
            for j in range(factor,n+1,factor):
                ans[j]=False
    temp=0
    for i in range(1,n+1):
        if ans[i]:
            temp+=1
            ans[i]=temp
    print(ans)
# ///////////////////////////////////////
def seivePrime(n):
    ans=[True for i in range(n+1)]
    for i in range(2,int((n**0.5)+1)):
        if ans[i]:
            for j in range(i*i,n+1,i):
                ans[j]=False
    arr=[]
    for i in range(2,n+1):
        if ans[i]:
            arr.append(i)
    return arr

def nFactorful(n,a,b):
    primes=seivePrime(b)
    diff=b-a
    ans=[0]*(diff+1)
    for i in primes:
        num=(a//i)*i
        if num<a:
            num+=i
        index=num-a
        for j in range(index,diff+1,i):
            ans[j]+=1
    count=0
    for i in ans:
        if i==n:
            count+=1
    print(count)
# ///////////////////////////////////////////
def seive(n):
    arr=[i for i in range(n+1)]
    arr[0]=arr[1]=0
    for i in range(2,int(n**(0.5))+1):
        for j in range(i*i,n+1,i):
            arr[j]=0
    ans=[]
    for i in range(len(arr)):
        if arr[i]!=0:
            ans.append(arr[i])
    return ans

def isFound(n,arr,start,end):
    if start>end:
        return False
    mid=(start+end)//2
    if arr[mid]==n:
        return True
    elif arr[mid]>n:
        return isFound(n,arr,start,mid-1)
    else:
        return isFound(n,arr,mid+1,end)
def primeTuple(n):
    primes=seive(n)
    count=0
    for j in primes:
        if isFound(2+j,primes,0,len(primes)-1):
            print(2,j)
            count+=1
    return count
# /////////////////////////////////////////////////////////


