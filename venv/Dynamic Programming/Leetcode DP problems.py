from math import inf,log
from collections import defaultdict,deque
from heapq import heapify,heappush as hpush,heappop as hpop,heapreplace as hreplace
import sys

def find_combination(string,index,output):

    if index==len(string):
        print(output)
        return

    find_combination(string,index+1,output+chr(96+int(string[index])))
    if index+1<len(string) and int(string[index:index+2])<27:
        find_combination(string,index+2,output+chr(96+int(string[index:index+2])))

# string="1123"
# find_combination(string,0,"")

















