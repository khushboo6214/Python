# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:54:57 2018

@author: khushi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
set path 
"""
dataset=pd.read_csv("50_Startups.csv")
dataset



x=dataset.iloc[:,:-1].values
x
y=dataset.iloc[:,4].values

y
#taking care of missing value
from sklearn.preprocessing import Imputer
# we remove missing value
#imputer=Imputer(missing_values="NaN",strategy="mean", axis=0)
# we are fitting
#imputer=imputer.fit(x[:,1:3])
# imputer.transform is to tranform all missing values
#x[:,1:3]=imputer.transform(x[:,1:3])
# solve dummy variable # label encoding for categorial data
from sklearn.preprocessing import LabelEncoder
labelencoder_state = LabelEncoder()
x[:,3]=labelencoder_state.fit_transform(x[:,3])
x
# france is 0 germany =2 spain is 1 #alphabetically

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
x
# same for Y variable as itscategorial 
#labelencoder_Profit = LabelEncoder()
#y=labelencoder_Profit.fit_transform(y)

"""# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc.x.fit_transform(x_train)
x_test=sc.x.transform(x_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)

"""


#Aviod Dummy Variable Trap
x=x[:,1:]# starting from 1 to end 
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#Import linear regression from sklearn.cross validation
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#plan is we build is class ,Model and Object are the instance of the class and method are use on an object to perform specific task
#Method is a tool we can use on an object to complete a specific task 
#Fit method to fit the method to the training set .this method is like afunction that will fit """
regressor.fit(x_train, y_train)
#now do regression on dependent and independent variable
y_pred =regressor.predict(x_test)

y_pred

#we made visualization for training

#we made visualization for predicted line

plt.scatter(x_train , y_train , color = "orange")

plt.plot(x_train,regressor.predict(x_train),color = "blue")
plt.title('Profit vs X(Training set)')
plt.xlabel('X')
plt.ylabel('Profit')
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
