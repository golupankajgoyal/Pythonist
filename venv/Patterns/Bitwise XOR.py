
# In a non-empty array of numbers, every number appears exactly twice except two numbers
# that appear only once. Find the two numbers that appear only once.
def find_two_unique_num(arr):
    xor_sum=0
    for num in arr:
        xor_sum^=num
    rightmost_set_bit=1
    while rightmost_set_bit&xor_sum==0:
        rightmost_set_bit=rightmost_set_bit<<1
    num1,num2=0,0
    for num in arr:
        if rightmost_set_bit&num==0:
            num1^=num
        else:
            num2^=num
    print(num1,num2)
# arr=[1, 4, 2, 1, 3, 5, 6, 2, 3, 5]
# # find_two_unique_num(arr)

# For a given positive number N in base-10, return the complement of its binary representation as a base-10 integer.
def find_compliment(n):
    msb=1
    temp=n
    while temp>0:
        msb=msb<<1
        temp=temp>>1
    msb-=1
    print(msb^n)
# find_compliment(10)

# Given a binary matrix representing an image, we want to flip the image horizontally, then invert it.
def flip_invert(mat):
    n=len(mat)
    for row in mat:
        for i in range((n+1)//2):
            row[i],row[n-i-1]=row[n-i-1]^1,row[i]^1
    print(mat)
mat=[
  [1,1,0,0],
  [1,0,0,1],
  [0,1,1,1],
  [1,0,1,0]
]
# flip_invert(mat)

