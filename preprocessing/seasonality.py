# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:25:57 2020

@author: anama
"""
#https://machinelearningmastery.com/time-series-seasonality-with-python/

import pandas as pd
from matplotlib import pyplot as plt
    
def difference(dataset, interval=1):
    """
    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    interval : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob

def calculate_interval(data_granularity, lag='W'):
    if lag == 'W': #week
        if 'min' in data_granularity:
             minutes_day = 24*60
             minutes_row = data_granularity.replace("min", '')
             minutes_row = int(minutes_row)
             minutes_week = minutes_day*7
             interval = round(minutes_week/minutes_row)
             
             
    if lag == "Y":
        if 'M' in data_granularity:
            interval = 12
        if 'D' in data_granularity:
            #does not consider leap years
            interval = 365
            
    return interval
def test_dataset():
    map_type = {"Global_active_power": 'str'}
    df = pd.read_csv('household_power_consumption.txt', sep=";", header=0, dtype=map_type)
    print(df.head())
    df = df.iloc[:201600, :]
    df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[["date", "Global_active_power"]]
    print(df["Global_active_power"].iloc[:5])
    df["Global_active_power"]  =  df["Global_active_power"].astype('str')
    print(df["Global_active_power"].iloc[:5])
    df.index = df['date']
    #df["Global_active_power"]  =  df["Global_active_power"].str.replace(r'\D', '.').astype(float)
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"],errors='coerce')
    
    print(df.head())
    df = df.resample('15min').mean()
   
    print(df.head())
    return df

def seasonal_adjustment(data=[], granularity='M', lag='Y'):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(['date'], drop=True, inplace=True)
    #print(data.head())
    #data = pd.read_csv('daily-min-temperatures.csv', header=0, index_col=0)
    print(data.index)
    print(data.columns)
    #data = test_dataset()
   
    X = data.values
    
    fig1, ax1 = plt.subplots()
    ax1.plot(data.index, X)
    plt.show()
    
    
    inter = calculate_interval(granularity, lag=lag)
    print("inter", inter)
    diff_values = difference(X, interval=inter)
    diff = pd.DataFrame({'value': diff_values})
    diff.index = data.index[inter:]
    print(diff[:5])
    fig2, ax2 = plt.subplots()
    ax2.plot(diff)
    plt.show()
    
    diff['date'] = diff.index.copy()
    diff.index = pd.RangeIndex(start=0, stop=diff.shape[0]) 
    return diff


