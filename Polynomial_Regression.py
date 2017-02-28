#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:35:33 2017

@author: DK
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing file
dataset = pd.read_csv("/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Polynomial_Regression/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values  #we need X to be a matrix for poly regression and not an array. Hence, we use a range of columns to specify set X
Y = dataset.iloc[:,2].values

#NOTE: The reason for a matrix is that every row will have 1 element eg for power 2 the array([2,3]) will produce 1,2,3,4,6,9.However, matrix(2,3) will produce 2 rows: 1,2,4 and 1,3,9 which is what we need.

#fitting linear regression to datatset
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X,Y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)          #creating a new X-set i.e independent variable set
poly_reg.fit(X_poly,Y)                      #not sure 

lin_reg_2 = LinearRegression()              
lin_reg_2.fit(X_poly, Y)                    #calculates the coefficients of the equation

#visualising linear regression
plt.scatter(X,Y,color = "red")
plt.plot(X, lin_reg_1.predict(X),color = "blue")
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#visualising polynomial regression
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = "blue")     #lin_reg_2 contains coefficients ; it doesn't know anything about polynomial regression,; so we have to apply it on X_poly. However, since we could have added new values in set X, we use the raw equation poly_reg.fit_transform to convert all values of set X in realtime
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#visualising polynomial regression with smoother curve
X_grid = np.arange(min(X), max(X),0.1)            #creates a vector with specifid range and step size
X_grid = X_grid.reshape((len(X_grid), 1)) 
plt.scatter(X,Y,color = "red")         #reshapes the vector to have only 1 column
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color = "blue")     #lin_reg_2 contains coefficients ; it doesn't know anything about polynomial regression,; so we have to apply it on X_poly. However, since we could have added new values in set X, we use the raw equation poly_reg.fit_transform to convert all values of set X in realtime
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()