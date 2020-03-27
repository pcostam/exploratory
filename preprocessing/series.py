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
import os
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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
   n_vars = 1 if type(data) is list else data.shape[1]
   df = pd.DataFrame(data)
   #n_vars = df.shape[1]
   print("nvars", n_vars)
 
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
   print("agg", agg)
   return agg
	
  
    
def preprocess_data(df, granularity, start_date, end_date):
    print("preprocess start date", type(start_date))
    minutes = str(granularity) + "min"
    df = select_data(df, start_date, end_date)
    print("downsample")
    df = downsample(df, minutes)

    return df 

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
    df = df_all.drop(['date'], axis = 1)
    yt = df['value'].tolist()
    decide_input(yt,16,1)
    
    data = series_to_supervised(df, n_in=n_input)
    return data


def create_dataset_as_supervised(table, sensorId, timesteps=3, limit=True, df_to_csv = False):
    df_all = pd.Dataframe()
    if df_to_csv == False:
        df_all = create_dataset(table, sensorId, limit=limit)
    if df_to_csv == True:
        df_all = csv_to_df(sensorId, limit = limit)
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

def select_data(df, min_date, max_date):  
    print("SELECT DATA 2")
    print("df.index", df.index)
    print("df.columns", df.columns)
    df['date'] = pd.to_datetime(df['date'])
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
   
    if ((min_date == 1) or (max_date == 1)):
        #df.reset_index(level=0, inplace=True)
        return df
    else:
        df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
        print("df.index", df.index)
        print("df.columns", df.columns)
        df.index = pd.RangeIndex(start=0, stop=df.shape[0])
        print("df.index", df.index)
        print("df.columns", df.columns)
        return df
    
def csv_to_df(sensorId, path, limit = True, n_limit=1000):
    
    if limit == True:
        df = pd.read_csv(path, nrows = n_limit)
    else:
        df = pd.read_csv(path)
        
    df[['value']] = df[['value']].astype('float32')
    return df

def find_gap(df, frequency):
     dates = df['date']
     res = []
     for index, tstamp in dates.items(): 
         try:        	
             tstamp1 = dates.loc[index]
             tstamp2 = dates.loc[index + 1]
             diff = tstamp1 - tstamp2
             diff_min = int(round(diff.total_seconds() / 60))
             if diff_min > 15:
                 #there is gap
                 print("gap", index)
                 res.append(index)
         except KeyError as e:
             pass
     return res
    
def csv_to_chunks(sensorId, path, limit=True, n_limit=1000):
    list_df = list()
    if limit == True:
        df = pd.read_csv(path, nrows = n_limit)
        df = downsample(df, '15min')
        print("after downsample", df)
        indexes = find_gap(df, 15)
        list_df = np.split(df, indexes)
        print("list df", list_df)
        #DO THIS IN INDEXES, IT'S MORE EASY!
        #TODO
        for i in range(0, len(list_df)):
            if list_df[i].shape[0] > 2000:
                size = list_df[i].shape[0]
                df_1 = list_df[i].iloc[:size,:]
                df_2 = list_df[i].iloc[size:,:]
                
                
      
     
    else:
        df = pd.read_csv(path)
        df = downsample(df, '15min')
        #list_df = np.array_split(df, 5)
        indexes = find_gap(df, 15)
        list_df = np.split(df, indexes)
        
        
    return list_df

def csv_to_TextFileReader(sensorId, path, limit=True, n_limit=1000):
    if limit == True:
        df_chunk = pd.read_csv(path, nrows = n_limit, chunksize=10000)
    else:
        df_chunk = pd.read_csv(path, chunksize=10000)
        
    return df_chunk
    
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



def script_to_csv():
    for sensorId in range(1,16):
        query = """
            SELECT ATG.date, STM.value
            FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
            WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND anomaly=0
            """ % (sensorId)
        path = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\infraquinta\\real\\normal\\sensor_"+ str(sensorId) + ".csv"
        
        generate_csv(query, sensorId, path)
        query = """
            SELECT ATG.date, STM.value
            FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
            WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND anomaly=1 
            """ % (sensorId)
        path = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\infraquinta\\real\\anomalies\\sensor_"+ str(sensorId) + ".csv"
        generate_csv(query, sensorId, path)
        
def generate_csv(query, sensorId, path):
     #import configuration 

     #root = configuration.read_config() 
     #db_config = configuration.get_db(root)
   
     sensor_id = 0
     
     mydb = mysql.connector.connect(host='localhost',user='root',password='banana')    
     
     df = pd.read_sql(query, con=mydb)
     df.to_csv(index=False, path_or_buf=path)
     
     print("  sensor " + str(sensorId) + ": " + str(df.shape[0]) + " rows")
        
     sensor_id += 1
      
     mydb.close()


def generate_anomalous(idSensor, limit=True, df_to_csv = False):
    df = pd.DataFrame()
    if df_to_csv: 
        init_path = os.path.dirname(os.getcwd())
        path = ""
        if init_path == "/content/drive/My Drive/Tese/exploratory":
              path =  init_path + "/wisdom/dataset/infraquinta/real/anomalies/sensor_"+ str(idSensor) + ".csv"
        else:
              path = init_path + "\\dataset\\infraquinta\\real\\anomalies\\sensor_" + str(idSensor) + ".csv"
    
        df = csv_to_df(idSensor, path, limit=limit)
    else:
        db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
        conn = create_engine(db_connection)
        query = ""
    
        if limit == False:
            query = """
             SELECT ATG.date, STM.value
            FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
            WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND anomaly=1
            """ % (idSensor)
        else:
            query = """
             SELECT ATG.date, STM.value
            FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
            WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND anomaly=1 limit 1000
            """ % (idSensor)
        
        df = pd.read_sql(query, conn)
    return df
 

   
def generate_normal(idSensor, limit=True, n_limit=1000, df_to_csv = False, to_chunks=True):
    df = pd.DataFrame()
    if df_to_csv: 
        init_path = os.path.dirname(os.getcwd())
        path = ""
        if init_path == "/content/drive/My Drive/Tese/exploratory":
            path =  init_path + "/wisdom/dataset/infraquinta/real/normal/sensor_"+ str(idSensor) + ".csv"
        else:
            path = init_path + "\\dataset\\infraquinta\\real\\normal\\sensor_" + str(idSensor) + ".csv"
     
        #df = csv_to_df(idSensor, path, limit=limit)
        if to_chunks == True:
            df = csv_to_chunks(idSensor, path, limit=limit, n_limit=n_limit)
        else:
            df = csv_to_df(idSensor, path, limit=limit, n_limit=n_limit)
        
    else:
        db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
        conn = create_engine(db_connection)
        query = ""
    
        if limit == False:
            query = """
             SELECT ATG.date, STM.value
            FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
            WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND anomaly=0
            """ % (idSensor)
        else:
            query = """
             SELECT ATG.date, STM.value
            FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
            WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND anomaly=0 limit 1000
            """ % (idSensor)
        
        df = pd.read_sql(query, conn)
        
    return df

def downsample(df, minutes):
     print("test 4")
     print("test downsample")
     print("minutes", minutes)
     print("df columns", df.columns)
     print("index", df.index)
     df['date'] = pd.to_datetime(df['date'])
     aux_1 = df['date'].copy()
     aux = df['date'].copy()
     df.index = aux
     df = df.resample(minutes).mean()
     print("df.index", df.index)
     print("df.columns", df.columns)
     print("aux 1", aux_1)
   
     df.index = pd.RangeIndex(start=0, stop=df.shape[0])
     df['date'] = aux_1
     print("df_date", df['date'])
     print("df index", df.index)
     print("df columns", df.columns)
     return df
 
def generate_sequences(sensorId, table, limit = True, df_to_csv = False):
    all_sequence = pd.DataFrame()
    normal_sequence = pd.DataFrame()
    anormal_sequence = pd.DataFrame()
    
    if df_to_csv == True:
        init_path = os.path.dirname(os.getcwd())
        path = ""
        if init_path == "/content/drive/My Drive/Tese/exploratory":
            path =  init_path + "/wisdom/dataset/infraquinta/real/sensor_"+ str(sensorId) + ".csv"
        else:
            path = init_path + "\\dataset\\infraquinta\\real\\sensor_" + str(sensorId) + ".csv"
        all_sequence = csv_to_df(sensorId, path, limit=limit)
        normal_sequence = generate_normal(sensorId, limit = limit, n_limit=129600, df_to_csv = df_to_csv)
        anormal_sequence = generate_anomalous(sensorId, limit=limit, df_to_csv = df_to_csv)
    
    else:
        all_sequence = create_dataset(table, sensorId, limit = limit)
        normal_sequence = generate_normal(sensorId, limit = limit, n_limit=129600, df_to_csv = df_to_csv)
        anormal_sequence = generate_anomalous(sensorId, limit=limit,df_to_csv = df_to_csv)
    
    return all_sequence, normal_sequence, anormal_sequence