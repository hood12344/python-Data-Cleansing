#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd

from tensorflow.keras.datasets import boston_housing

from sklearn.metrics import mean_squared_error, mean_absolute_error



import myfun

classes = ['housing_median_age','total_rooms','total_bedrooms','population','households']

train_x, test_x, train_y, test_y,scaler=\
              myfun.ML_read_dataframe_標準化("housing.xlsx",
              classes,
              ['median_income'])




from sklearn import preprocessing
# 標準化


model =  tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512,
                                activation='tanh',
                                input_shape=[train_x.shape[1]]))
model.add(tf.keras.layers.Dense(512, activation='tanh'))
model.add(tf.keras.layers.Dense(1))


model.compile(loss='mse',
              optimizer='sgd',
              metrics=['mae'])

history=model.fit(train_x, train_y,
          epochs=200,
          batch_size=10000)

#保存模型架構
with open("model_Boston.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model_Boston.h5")


# testing
print("start testing")
cost = model.evaluate(test_x, test_y)
print("test cost: {}".format(cost))

Y_pred2 = model.predict(test_x)  # Y predict

print(Y_pred2[:10])
print(test_y[:10])
# 印出測試的結果
Y_pred = model.predict(test_x)
print("預測:",Y_pred )
print("實際:",test_y)
print('MAE:', mean_absolute_error(Y_pred, test_y))
print('MSE:', mean_squared_error(Y_pred, test_y))

