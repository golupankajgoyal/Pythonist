import matplotlib.pyplot as plt
import numpy as np
import statistics as st
# %matplotlib inline

Xj= [0,1,2,3,4,5]
Yk=[0,2,3,4,6,8]
delw=[]
n=1
for num1, num2 in zip(Xj, Yk):
   delw.append(n*num1*num2)
plt.plot(Yk,delw)
plt.title("Activity Product Rule")
plt.xlabel("Yk")
plt.ylabel("del(Wkj)")
plt.show()

Wkj=[]
Wkj.append(0)
for i in range(1,len(delw)):
  Wkj.append(Wkj[i-1]+delw[i-1])
plt.plot(Yk,Wkj)
plt.title("Activity Product Rule")
plt.xlabel("Yk")
plt.ylabel("Wkj")
plt.show()

c=0.01
delw1=[]
for num1, num2, num3 in zip(Yk,Xj,Wkj):
  delw1.append(n*num1*(num2 - c*num3))
plt.plot(Yk,delw1)
plt.title("Generalised Activity Product Rule")
plt.xlabel("Yk")
plt.ylabel("del(Wkj)")
plt.show()

x_avg= st.mean(Xj)
y_avg= st.mean(Yk)
delw2=[]
for num1, num2 in zip(Xj, Yk):
  delw2.append(n*(num1-x_avg)*(num2-y_avg))
plt.plot(Yk,delw2)
plt.title("Covariance Hypothesis")
plt.xlabel("Yk")
plt.ylabel("del(Wkj)")
plt.show()



