# r w a read readline write
f = open('data','r')
# print(f.readline())
# print(f.readline(4))

f1 = open('data1','w')
# f1.write("Something")
# f1.write("Pankaj")

# for data in f:
#     f1.write(data)
    # print(data)

# put all data into list
# listData = f.readlines()
# print(listData)

# write all data from list into file
# l = ['Hii' , 'A', 'B', 'C', 'D', 'E']
# f1.writelines(l)

data = f.read()
f.seek(3)
for i in f:
    print(i)



