
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

df = pd.read_excel("housing-2.xlsx",0) #('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(df.columns)

"""
 #longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value	ocean_proximity

"""
"""print("===========資料清洗Data Cleaning==============")
print("====3-1.資料清洗   異常值==============")

print("異常值",df.applymap(np.isreal))
print(df.applymap(np.isreal).all(1))
print("異常值row:",df[~df.applymap(np.isreal).all(1)])
row1=df[~df.applymap(np.isreal).all(1)]"""


"""print("====3-2.資料清洗   異常值==============")
#df =pd.to_numeric(df.columns , errors ='coerce').fillna(999).astype('float')
# df['ALP']=df['ALP'].fillna(0)
#df['ALP2'] = np.where(df['ALP'].astype(str).fillna('0').str.contains('ALP'),df.ALP, None)
#df['ALP3'] = df['ALP2'].ffill()"""


"""print("====3-3.0資料清洗 資料轉換==============")

df['Date'] = pd.to_datetime(df['Date'])

print(df.head())
print(df.dtypes )"""
"""print("====3-2 資料清洗 移除row==============")
df.dropna(subset=['Date'], inplace = True)"""


"""print("====3-3 資料清洗 修改>120==============")


for x in df.index:
  if df.loc[x, "CHE"] > 120:
    df.loc[x, "CHE"] = 120


print("====3-4 資料清洗 移除>120  的row==============")
for x in df.index:
  if df.loc[x, "CHOL"] > 120:
    df.drop(x, inplace = True)"""


"""print("====3-5 資料清洗 移除 重複資料  的row==============")
print("重複資料:",df.duplicated())
print("重複資料的那一筆:",df[df.duplicated()])
print("重複資料的那一筆的index:",df[df.duplicated()].index)
x=df[df.duplicated()].index
df.drop(x, inplace=True)"""


"""print("====3-3 資料清洗 修改==============")


df.loc[0, 'ALB'] = 45
print(df.head(10))
exit()"""








print("是否有 NAN ：",df.isnull().values.any())
print("NAN 的數量：",df.isnull().sum())

print(df.isnull().sum().sum())
print(df.isnull().values.any())
rows_with_nan = [index for index, row in df.iterrows() if row.isnull().any()]
print("NAN 的index",rows_with_nan)

"""print("====2-1 處理缺失值 使用特殊碼 999==============")
df['ALB'] = df['ALB'].fillna(999)
df['ALP'] = df['ALP'].replace(np.nan,-999)
print("====2-2 處理缺失值 使用均值==============")
print("====2-3 處理缺失值 使用均值數==============")
mean1=df['ALT'].mean()        #均值
df['ALT'] = df['ALT'].fillna(mean1)
print("====2-4 處理缺失值 使用中位數==============")
median1=df['AST'].median()      #中位数
df['AST'] = df['AST'].fillna(median1)
print("====2-5 處理缺失值 使用眾數==============")
mode1=df['BIL'].mode()
df['BIL'] = df['BIL'].fillna(mode1)
print("====2-6 處理缺失值 刪除row==============")
df = df.drop(np.nan, axis=0)
# df = df.drop(np.nan, axis=1)
#df["CHE"]=df.dropna(subset = ["CHE"], inplace=True)
df = df.dropna(axis=0)

print("====2-7 處理缺失值 刪除col==============")
# Drop all columns with NaN values
df=df.dropna(axis=1)
"""
df = df.dropna(axis=0)
print("是否有 NAN ：",df.isnull().values.any())
def Array_ToDic(lst):
  res_dct = {lst[i]: i for i in range(0, len(lst), 1)}
  return res_dct

def pandas_col_StingToInt(df, colName):
  # df["target2"] = df['status'].map({'良好':1,'普通':0})
  return df[colName].map(Array_ToDic(df[colName].unique()))


df["ocean_proximity"] = pandas_col_StingToInt(df, "ocean_proximity")
"""pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)"""
print(df.head())
df.to_excel("housing-3.xlsx")