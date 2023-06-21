# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:50:05 2023

@author: MRUTYUNJAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Datascience Classes\10th april\student_info.csv")
df.head()
df.tail()
df.shape
df.info()
df.describe()
plt.scatter(x=df.study_hours,y=df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Student marks")
plt.title("Scatter Plot of Students Study Hours vs Student marks")
plt.show()
df.isnull().sum()
df.mean()
df2=df.fillna(df.mean())
df2.isnull().sum()
df2.head()
X=df2.drop("student_marks",axis="columns")
y=df2.drop("study_hours",axis="columns")
print("shape of X=", X.shape)
print("shape of y=", y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=51)
print("shape of X_train=",X_train.shape)
print("shape of y_train=",y_train.shape)
print("shape of X_test=",X_test.shape)
print("shape of y_test=",y_test.shape)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_
m=3.93
c=50.44
y=m*4+c
y
lr.predict([[4]])[0][0].round(2)
y_pred=lr.predict(X_test)
y_pred
pd.DataFrame(np.c_[X_test, y_test, y_pred], columns=  ["study_hours","student_marks_original","student_marks_predicted"])
lr.score(X_test,y_test)
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.plot(X_train,lr.predict(X_train),color="r")
import joblib
joblib.dump(lr,"student_marks_predictor.pkl")
model=joblib.load("student_marks_predictor.pkl")
model.predict([[5]])[0][0]
