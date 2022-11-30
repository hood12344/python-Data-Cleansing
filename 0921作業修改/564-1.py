#!/usr/bin/python
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics


print("====讀取資料==============")

df = pd.read_excel("C肝-資料清洗1-文字轉數字.xlsx",0) #('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(df.columns)




print("===========資料清洗Data Cleaning==============")
print("====1.資料清洗 文字轉數字==============")
def Array_ToDic(lst):
  res_dct = {lst[i]: i for i in range(0, len(lst), 1)}
  return res_dct

def pandas_col_StingToInt(df,colName):
    # df["target2"] = df['status'].map({'良好':1,'普通':0})
    return  df[colName].map(Array_ToDic(df[colName].unique()))


df["Category2"] = pandas_col_StingToInt(df,"Category")
df["Category2"] = pandas_col_StingToInt(df,"Category")
df["Sex2"] =df["Sex"].replace({'m': 1, 'f': 0})


print(df.head())