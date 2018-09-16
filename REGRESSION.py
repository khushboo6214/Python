# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:54:57 2018

@author: khushi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
 os.getcwd()
"""
set path 
"""
dataset=pd.read_csv("Salary_Data.csv")
dataset

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
x
y
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#Import linear regression from sklearn.cross validation
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#plan is we build is class ,Model and Object are the instance of the class and method are use on an object to perform specific task
#Method is a tool we can use on an object to complete a specific task 
#Fit method to fit the method to the training set .this method is like afunction that will fit """
regressor.fit(x_train, y_train)
#now do regression on dependent and independent variable
y_pred =regressor.predict(x_test)



#we made visualization for training
plt.scatter(x_train,y_train,color = "orange")
#we made visualization for predicted line
plt.plot(x_train,regressor.predict(x_train),color = "blue")
plt.title('salary vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('salary')
plt.show()

#Visualizing the test set Results
plt.scatter(x_test,y_test,color = "red")
#we made visualization for predicted line
# the blue line predicted line created is on train data set we will keep this as it is
plt.plot(x_train,regressor.predict(x_train),color = "blue")
plt.title('salary vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('salary')
plt.show()


# machine is simple linear regression model learning on training data set to do prediction
"""

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
"""
