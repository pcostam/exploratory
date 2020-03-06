# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:39:51 2019

@author: anama
"""
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
   """
   Frame a time series as a supervised learning dataset.
   Arguments:
	data: Sequence of observations as a list or NumPy array.
	n_in: Number of lag observations as input (X).
	n_out: Number of observations as output (y).
	dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
	Pandas DataFrame of series framed for supervised learning.
   """
   #n_vars = 1 if type(data) is list else data.shape[1]
   #df = pd.DataFrame(data)
   n_vars = df.shape[1]
 
   cols, names = list(), list()
    
   # input sequence (t-n, ... t-1)
   for i in range(n_in, 0, -1):
       cols.append(df.shift(i))
       names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
   for i in range(0, n_out):
       cols.append(df.shift(-i))
       if i == 0:
           names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
       else:
           names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
   agg = pd.concat(cols, axis=1)

   agg.columns = names
	# drop rows with NaN values
   if dropnan:
       agg.dropna(inplace=True)
   return agg
	
    
def plot(y):
    #x = np.arange(1,15)
    plt.plot(y)
  
def resampler():
    return True

def create_data(table, sensorId, n_input, limit=False):
    df_all = create_dataset(table, sensorId, limit=limit)
    y = df_all['value'][:14]
    #plot(y)
    print(y)
    #plot(df['value'][:14])
    #date = df['date'].tolist()
    print("DF>>>>>>>>>>", df_all.columns.values)
    df = df_all.drop(['date','sensortgId', 'id'], axis = 1)
    yt = df['value'].tolist()
    decide_input(yt,16,1)
    
    data = series_to_supervised(df, n_in=n_input)
    return data


def create_dataset_as_supervised(table, sensorId, timesteps=3, limit=True):
    df_all = create_dataset(table, sensorId, limit=limit)
    y = df_all['value'][:14]
    #plot(y)
    print(y)
    #plot(df['value'][:14])
    #date = df['date'].tolist()
    print("DF>>>>>>>>>>", df_all.columns.values)
    df = df_all.drop(['date','sensortgId', 'id'], axis = 1)
    yt = df['value'].tolist()
    decide_input(yt,16,1)
    data = series_to_supervised(df, timesteps)
    data["date"] = df_all["date"]
    print("data", data)
    print("DATA>>>>>>>", data.shape[1])
    X = np.array(data.iloc[:,:timesteps])
    print("X>>>>>>", X)
    y = np.array(data.iloc[:,timesteps])
    print("y>>>>", y)
    X = pd.DataFrame(X, columns=["var(t-3)", "var(t-2)", "var(t-1)"])
    X["date"] = np.array(data["date"])
    y = pd.DataFrame(y, columns=["value"])
    y["date"] = np.array(data["date"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
    return X_train, X_test, y_train, y_test
    
def create_dataset(table, sensorId, limit = True):    
    db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    conn = create_engine(db_connection)
    if "sensortmmeasure" in table:
        if limit:
            query = "SELECT date, value FROM " + table + " where sensortmId="+ sensorId + " limit 1000"
        else:
            query = "SELECT date, value FROM " + table + " where sensortmId="+ sensorId 
    
    if "sensortgmeasure" or "sensortgmeasurepp" in table:
          if limit:
            query = "SELECT date, value FROM " + table + " where sensortgId="+ sensorId + " limit 1000"
          else:
            query = "SELECT date, value FROM " + table + " where sensortgId="+ sensorId 
    
    df = pd.read_sql(query , conn)
    
   
    return df

def date_N_days_ago(date, days):
    return date - datetime.timedelta(days=days)
    
def correlation_time_series_and_lag():
    return True

def decide_input(yt, max_lag_period, forecast_period):
    """
    Based on correlation, decide the lag period (input for neural network)
    Arguments: 
    yt: univariate time series(attribute) as list of values
    max_lag_period: maximum lag period to be tested. Max value is the size of yt-1
    forecast_period: timestep to predict 
    Returns: list of appropriate lag periods to use in neural networks
    """ 
    correlations = list()
    yt_mean = np.mean(yt)

    #correlations of time series yt at lag k (yt-1...yt-max_lag_period+1)
    for k in range(1, max_lag_period + 2):
        yt_k = shift(yt, k, cval=np.NaN)
        a = yt-yt_mean
        a = np.nan_to_num(a)
        b = yt_k-yt_mean
        b = np.nan_to_num(b)
        print(np.dot(a, b))
        r_k = np.sum(np.dot(a, b))
        correlations.append(r_k)
        
    print("CORRELATIONS", correlations)
    
    s = list()
    muls = list()
    for i in range(0, len(correlations)-1):
        for j in range(0, len(correlations)-1):
            if(j != i):
                muls.append(abs(correlations[i-j]))
        
        s.append(abs(correlations[i + forecast_period])/abs(np.prod(muls)))
    
    print(s)
    lag_periods = np.argpartition(s, -max_lag_period)[-max_lag_period:] + 1
    
    print("lag_periods", lag_periods)
    return lag_periods

def generate_anomalous(idSensor, limit=True):
    db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    conn = create_engine(db_connection)
    if limit == False:
        query = """
        SELECT ATG.date, STM.value
        FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
        WHERE idmeasurestg=STM.id and anomaly=1
        """ 
    else:
        query = """
        SELECT ATG.date, STM.value
        FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
        WHERE idmeasurestg=STM.id and anomaly=1 limit 1000
        """ 
        
    df = pd.read_sql(query, conn)
    return df
def generate_normal(idSensor, limit=True):
    db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    conn = create_engine(db_connection)
    query = ""
    if limit == False:
        query = """
         SELECT ATG.date, STM.value
        FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
        WHERE idmeasurestg=STM.id and anomaly=0
        """ 
    else:
        query = """
         SELECT ATG.date, STM.value
        FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
        WHERE idmeasurestg=STM.id and anomaly=0 limit 1000
        """ 
        
    df = pd.read_sql(query, conn)
    return df

def generate_sequences(sensorId, table, limit = True):
    all_sequence = create_dataset(table, sensorId, limit = limit)
    return all_sequence, generate_normal(sensorId, limit = limit), generate_anomalous(sensorId, limit=limit)