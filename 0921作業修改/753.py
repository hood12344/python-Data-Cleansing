#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import California Housing Prices

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

classes = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
print(classes)

data = pd.DataFrame(x_train, columns=classes)
print(data.head())


data['MEDV'] = pd.Series(data=y_train)
print(data.head())
print(data.describe())


data.to_csv("boston.csv", sep='\t')
writer = pd.ExcelWriter('boston.xlsx', engine='xlsxwriter')
data.to_excel(writer, sheet_name='Sheet1',index=False)
writer.save()