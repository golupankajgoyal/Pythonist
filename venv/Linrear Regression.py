

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
data=pd.read_csv("Salary_Data_LR.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
lireg=LinearRegression()
lireg.fit(x_train,y_train)
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,lireg.predict(x_train),color="red")
plt.title("Salary Exp in Train")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,lireg.predict(x_test),color="green")
plt.title("Salary Exp in Train")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()




