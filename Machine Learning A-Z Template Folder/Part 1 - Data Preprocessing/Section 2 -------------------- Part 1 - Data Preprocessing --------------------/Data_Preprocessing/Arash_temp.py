#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:47:43 2017

@author: Arash
"""

# Machine learning A-z with Pthon and R
# Arash Nouri
# Chapter 1
# Data template

# Loading tools________________________________________________________________
import numpy as np
import pandas as pd


# Loading data_________________________________________________________________
data = pd.read_csv('Data.csv')
print(                                                                        )
print('data:-----------------------------------------------------------------')
print(data)
print('----------------------------------------------------------------------')

print('data size:------------------------------------------------------------')
print(data.size)
print('----------------------------------------------------------------------')

print('data shape:-----------------------------------------------------------')
print(data.shape)
print('----------------------------------------------------------------------')

print('data description:-----------------------------------------------------')
print(print(data.describe(include = 'all').transpose))
print('----------------------------------------------------------------------')

print('Missed data:----------------------------------------------------------')
print(print(data.isnull().sum()))
print('----------------------------------------------------------------------')

# Slicing the data-------------------------------------------------------------
X = data.iloc[:,:-1].values     # Inputs

y = data.iloc[:,3].values       # outputs


# Missing values_______________________________________________________________
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values = 'NaN', 
                   strategy = 'mean', 
                   axis = 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Categprical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Splitting into trainig and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)


