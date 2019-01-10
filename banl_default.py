# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 08:40:38 2018

@author: TEJA
"""

import pandas as pd
import numpy as np
train=pd.read_csv(r"C:\Users\TEJA\Desktop\train_indessa.csv")
test=pd.read_csv(r"C:\Users\TEJA\Desktop\test_indessa.csv")
data=train.append(test)
data.reset_index(inplace=True)
data.info()
z=pd.get_dummies(data.application_type)
data=pd.concat([data,z],axis=1)
data["term"]=data["term"].replace({"36 months":36*4,"60 months":60*4})
data.loc[data["member_id"]<9999999,"member_id"]=0
data.loc[data["member_id"]>9999999,"member_id"]=1
data["batch_enrolled"]=data.batch_enrolled.replace({' ':0,'NaN':0})
data["batch_enrolled"]=data["batch_enrolled"].fillna(0)
data.loc[data["batch_enrolled"]!=0,'batch_enrolled']=1
data["grade"].value_counts()
import re
data["last_week_pay"]
for i in data["last_week_pay"]:
    i=i.split(' ')[0]
    

data["last_week_pay"]=data["last_week_pay"].map(lambda i:i.split(' ')[0].strip('th') )
data["last_week_pay"]=data["last_week_pay"].fillna(0)
data["last_week_pay"]=data["last_week_pay"].map(lambda i:int(i))

df_temp=data.groupby('sub_grade').agg({'loan_amnt':'mean'}).rename(columns={'loan_amnt':'sg_mean'}).reset_index()
df_temp
df_temp2=data.groupby('grade').agg({'loan_status':'sum'})
df_temp2
np.shape(train)
data["emp_title"].unique()
data["emp_title"].value_counts()
data["loan_status"].value_counts()
col_list=data.columns
col_list
col_obj=[]
obj_list=data.dtypes.values
obj_list
for col,obj in zip(col_list,obj_list):
    if obj==object:
        col_obj.append(col)
col_obj.remove("addr_state")    
col_obj.remove("application_type")
col_obj
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["application_type"].head()
data[col_obj].count()
np.shape(data)

col=["grade","home_ownership"]
for i in col:
    z=pd.get_dummies(data[i])
    data=pd.concat([data,z],axis=1)
np.shape(data)
