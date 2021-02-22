import sys
from collections import defaultdict
# For KMP Algo(longest prefix suffix)
def findLpsArr(string):
    ans=[0]*len(string)
    i=1
    j=0
    while i<len(string):
        if string[j]==string[i]:
            ans[i]=j+1
            j+=1
            i+=1
        else:
            if j!=0:
                j=ans[j-1]
            else:
                ans[i]=0
                i+=1
    return ans

def kmpSearch(string,pattern):
    lenPat=len(pattern)
    lenStr=len(string)
    lps=findLpsArr(pattern)
    i=j=0
    while i<lenStr and j<lenPat:
        if string[i]==pattern[j]:
            i+=1
            j+=1
        else:
            if j!=0:
                j=lps[j-1]
            else:
                i+=1
    if j==lenPat:
        return True
    else:
        return False

def zAlgo(string,pattern):
    string=pattern+"$"+string
    lenPattern=len(pattern)
    lenString=len(string)

    size=len(string)
    arr=[0]*size
    l=r=0
    i=1
    while(i<size-1):
        if i>r:
            l=r=i
            while r<size and string[r]==string[r-l]:
                r+=1
            arr[i]=r-l
            r-=1
        else:
            k=i-l
            if arr[k]+i<r:
                arr[i]=arr[k]
            else:
                l=i
                while r<size and string[r-l]==string[r]:
                    r+=1
                arr[i]=r-l
                r-=1
        i+=1
    print(arr)
    for i in range(lenString):
        if lenPattern==arr[i]:
            print(i-lenPattern)

def findMaxPalindrom(string,n):
    maxLen=0
    l=r=0
    for i in range(n):
        l=i-1
        r=i+1
        while l>=0 and r<n and string[l]==string[r]:
            l-=1
            r+=1
        if l!=i-1:
            maxLen=max(maxLen,r-l-1)
        l=i
        r=i+1
        while l>=0 and r<n and string[l]==string[r]:
            l-=1
            r+=1
        if l!=i:
            maxLen=max(maxLen,r-l-1)
    print(maxLen)
# string="abaababaaabababa"
# findMaxPalindrom(string,len(string))

def findLargestStringOfKChar(string,k):
    maxLen=0
    winStart=0
    count=0
    freq=defaultdict(int)
    for winEnd in range(len(string)):
        if freq.get(string[winEnd],None):
            freq[string[winEnd]]+=1
        else:
            freq[string[winEnd]]=1
            count+=1
        while count>k:
            freq[string[winStart]]-=1
            if freq[string[winStart]]==0:
                count-=1
            winStart+=1
        maxLen=max(maxLen,winEnd-winStart+1)
    print(maxLen)

# string="araaci"
# k=2
# findLargestStringOfKChar(string,k)

def longestStringWithoutRepeating(string):
    winStart=0
    lastIndex=defaultdict(int)
    maxLen=0
    for winEnd in range(len(string)):
        char=string[winEnd]
        if char in lastIndex:
            winStart=max(winStart,lastIndex[char]+1)
        lastIndex[char]=winEnd
        maxLen=max(maxLen,winEnd-winStart+1)
    print(maxLen)
# String="aabccbb"
# longestStringWithoutRepeating(String)

# longestStringWithSameLetterAfter k letter Replacement in the string
def stringWithReplace(string,k):
    winStart=0
    maxFreqInWin=0
    maxLen=0
    freq=defaultdict(int)
    for winEnd in range(len(string)):
        char=string[winEnd]
        if char in freq:
            freq[char]+=1
        else:
            freq[char]=1
        maxFreqInWin=max(maxFreqInWin,freq[char])
        print(winEnd,winStart,maxFreqInWin,freq)
        if winEnd-winStart+1-maxFreqInWin>k:
            freq[string[winStart]]-=1
            winStart+=1
        maxLen=max(winEnd-winStart+1,maxLen)
    print(maxLen)

# String="abccde"
# k=1
# stringWithReplace(String,k)

# Longest SubArray with Ones after Replacement
def stringWithAllOnes(string,k):
    winStart=0
    maxLen=0
    maxcountOne=0
    for winEnd in range(len(string)):
        if string[winEnd]==1:
            maxcountOne+=1
        if winEnd-winStart+1-maxcountOne>k:
            if string[winStart]==1:
                maxcountOne-=1
            winStart+=1
        maxLen=max(maxLen,winEnd-winStart+1)
        print(winEnd,winStart)
    print(maxLen)

# string=[0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
# k=2
# stringWithAllOnes(string,k)

# Write a function to return a list of starting indices of the anagrams of the pattern in the given string.
def findPermutation(string,pattern):
    frq=defaultdict(int)
    winStart=0
    pLen=len(pattern)
    distinctChar=0
    matchChar=0
    for char in pattern:
        if frq.get(char,None):
            frq[char]+=1
        else:
            frq[char]=1
            distinctChar+=1
    for winEnd in range(len(string)):
        rChar=string[winEnd]
        if rChar in frq:
            frq[rChar]-=1
            if frq[rChar]==0:
                matchChar+=1
        if winEnd-winStart+1>pLen:
            lChar=string[winStart]
            if lChar in frq:
                if frq[lChar]==0:
                    matchChar-=1
                frq[lChar]+=1
            winStart+=1
        if matchChar==distinctChar:
            print(winStart,end=" ")
# String="abbcabc"
# Pattern="abc"
# findPermutation(String,Pattern)

# Given a string and a pattern, find the smallest substring in the given string which has all the characters of the given pattern.

def smallestSubString(string,pattern):
    winStart=0
    frq=defaultdict(int)
    minLen=sys.maxsize
    charCount=0
    substrStart=-1
    for char in pattern:
        if char in frq:
            frq[char]+=1
        else:
            frq[char]=1

    for winEnd in range(len(string)):
        rChar=string[winEnd]
        if rChar in frq:
            frq[rChar]-=1
            if frq[rChar]>=0:
                charCount+=1
        while charCount==len(pattern):
            if minLen>winEnd-winStart+1:
                minLen=winEnd-winStart+1
                substrStart=winStart
            lChar=string[winStart]
            winStart+=1
            if lChar in frq:
                if frq[lChar]==0:
                    charCount-=1
                frq[lChar]+=1
    if substrStart!=-1:
        print(string[substrStart:substrStart+minLen])
    else:
        print("Not Found")

# String="aabdec"
# Pattern="abc"
# smallestSubString(String,Pattern)

# Given an array arr of unsorted numbers and a target sum,
# count all triplets in it such that arr[i] + arr[j] + arr[k] < target where i, j, and k are three different indices.
# Write a function to return the count of such triplets.
def findTriplet(arr,target):
    arr=sorted(arr)
    print(arr)
    triplets=[]
    for i in range(len(arr)):
        left=i+1
        right=len(arr)-1
        while left<right:
            diff=target-arr[i]-arr[left]-arr[right]
            if diff>0:
                temp=right
                while temp!=left:
                    triplets.append((arr[i],arr[left],arr[temp]))
                    temp=temp-1
                left+=1
            else:
                right-=1
    print(triplets)

# arr=[-1, 4, 2, 1, 3]
# target=5
# findTriplet(arr,target)

def applyBackslash(string):
    output=""
    for char in string:
        if char=="#":
            output=output[:-1]
        else:
            output=output+char
    print(output)


def printString(string):
    i=len(string)-1
    backslash=0
    while i>=0:
        if string[i]=="#":
            backslash+=1
        elif backslash>0:
            backslash-=1
        else:
            print(string[i],end="")
        i-=1
# String="abc##d#ef"
# # applyBackslash(String)
# printString(String)
