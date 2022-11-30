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

classes = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']

train_x, test_x, train_y, test_y,scalerX,scalerY=\
              myfun.ML_read_dataframe_標準化xy("housing-3.xlsx",
              classes,
              ['median_house_value'])

#train_y=train_y/100
#test_y=test_y/100
#train_y = scaler.transform(train_y.reshape(1, -1) ).reshape(-1, 1)  # 把資料轉換
#test_y = scaler.transform(test_y)  # 把資料轉換



from sklearn import preprocessing
# 標準化


model =  tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512,
                                activation='tanh',
                                input_shape=[train_x.shape[1]]))
model.add(tf.keras.layers.Dense(512, activation='tanh'))
model.add(tf.keras.layers.Dense(1))


opti1=tf.keras.optimizers.Adam(lr=0.001)    # 使用Adam 移動 0.001  #  內定值 learning_rate=0.001,
model.compile(loss='mse',
              #optimizer='sgd',
              optimizer=opti1,
              metrics=['mae'])

history=model.fit(train_x, train_y,
          epochs=200,
          batch_size=1000)

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
Y_pred3=Y_pred2.reshape(Y_pred2.shape[0],1) # 1D 轉 2D
Y_pred3 = scalerY.inverse_transform(Y_pred3)   # 標準化還原
Y_pred3=Y_pred3.reshape(Y_pred3.shape[0]) # 2D 轉 1D

test_y3=scalerY.inverse_transform(test_y.reshape(test_y.shape[0],1)).reshape(test_y.shape[0])

print(Y_pred2[:10])
print(test_y[:10])
# 印出測試的結果
Y_pred = model.predict(test_x)
print("預測:",Y_pred )
print("實際:",test_y)
print("預測標準化還原的答案:",Y_pred3 )
print("實際標準化還原的答案::",test_y3)
print('MAE:', mean_absolute_error(Y_pred, test_y))
print('MSE:', mean_squared_error(Y_pred, test_y))

