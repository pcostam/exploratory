# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:29:21 2020

@author: anama
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from preprocessing.series import create_dataset, create_dataset_as_supervised
import numpy as np
import pandas as pd

def plot(y_test, y_pred):
    order = np.arange(330)
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(order, y_test, color='b')
    plt.plot(order, y_pred, color='r')
    plt.show()
def do_SVR(X, y):
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)
    return regressor
def test():
    Xtrain, Xtest, ytrain, ytest = create_dataset_as_supervised("sensortgmeasurepp", "12", limit=True)
    
    print("Xtrain", Xtrain)
  
    y_test = np.array(ytest["value"]).reshape(-1,1)
    y_train = np.array(ytrain["value"]).reshape(-1,1)
    X_train = Xtrain.loc[:,["var(t-3)", "var(t-2)", "var(t-1)"]]
    X_test = Xtest.loc[:, ["var(t-3)", "var(t-2)", "var(t-1)"]]
   
    ssX = StandardScaler()
    ssY = StandardScaler()
    X_train = ssX.fit_transform(X_train)
    y_train = ssY.fit_transform(y_train)
   
    model = do_SVR(X_train, y_train.ravel())
    
    y_pred = model.predict(X_test)
    y_pred = ssY.inverse_transform(y_pred)
    print(y_pred)
    
    y_test = y_test.ravel()
    plot(y_test, y_pred)
      
    df = create_dataset("sensortgmeasurepp", "12", limit=True)
    
    #janelas de 1 hora
    df["avg_time_day"] = df["value"].rolling(60, min_periods=1).mean()
    print(df)
   
    tol = 0.7
    for i in range(0, y_pred.shape[0]):
        avg = df["avg_time_day"][i]
       
        res = abs(y_pred[i] - y_test[i])/avg
        print(res)
     
       
        if res > tol:
            print("anomaly in i", i)
            
    return True
    