# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import input file
#data=pd.read_csv('salary.csv')
data = pd.read_csv('https://raw.githubusercontent.com/suyogyaman/simpleLinearRegression/master/salary.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values
z = 11.00
z = z.reshape(-1,1)

#Splitting data set into Training set and Test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)


#Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred_15 = regressor.predict([[15]]) #Lets predict salary for 15 years experience

#Visualizing the Training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')

#Visualizing the Test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test set))')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()