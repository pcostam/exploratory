# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:42:40 2020

@author: anama
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:37:08 2020

@author: anama
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from preprocessing.series import create_dataset, create_dataset_as_supervised

def plot(X_train, X_outliers):
    order = np.arange(1000)
    
    fig, ax = plt.subplots(figsize=(10,6))
   

    ax.plot(order, X_train, color='blue')
    
    ax.scatter(X_outliers['order'],X_outliers['measures'], color='red')
    plt.show()
  
def do_SVM(X_train):
    nu = 0.05  
    ocsvm = OneClassSVM(kernel='rbf', gamma=0.05, nu=nu)
    ocsvm.fit(X_train)
    pred = ocsvm.predict(X_train)
    print("pred", pred)
    measure_outliers = X_train[pred == -1]
    measure_outliers = pd.Series(measure_outliers[:,0], name="measures")
 
    idx = pd.Series(np.where(pred == -1)[0], name="order")
    frames = [idx, measure_outliers]
    X_outliers = pd.concat(frames, axis=1, sort=False)
    print(X_outliers)
    return X_outliers
  
    

    
def test():
    df_pressure = create_dataset("sensortgmeasure", "11")
    df_pressure = df_pressure.rename(columns={"value": "pressure"})
    df_pressure = df_pressure.drop(['date','sensortgId', 'id'], axis = 1)
    df_flow = create_dataset("sensortgmeasure", "12")
    df_flow = df_flow.drop(['date','sensortgId', 'id'], axis = 1)
    df_flow = df_flow.rename(columns={"value": "flow"})

 
    frames = [df_pressure, df_flow]
    Xtrain = pd.concat(frames, axis=1, sort=False)
    

    
    #normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(Xtrain)
    
    X_outliers = do_SVM(X_train)
    
    plot(X_train[:,0], X_outliers)
    
  
 
   
    return True