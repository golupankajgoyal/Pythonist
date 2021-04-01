from itertools import permutations
# l = list(permutations(range(1, 4)))
for i in permutations(range(1, 4)):
    temp=list(i)
    print(type(temp),type(i))
# print(l)
