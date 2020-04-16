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
#https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, stride=None):
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
 
   cols, names, pivots = list(), list(), list()
    
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

   
   #stride - delete windows
   if stride != None:
       indexes_to_drop = list()
       for i in range(stride, agg.shape[0], stride):
           print("index", i)
           pivots += [i]
           
       onset = 0
       offset = pivots[0]
       for i in range(0, len(pivots)):
           print("onset", onset)
           print("offset", offset)
           to_drop = [ x for x in range(onset,offset)]
           indexes_to_drop += to_drop
           try:
               onset = pivots[i] + 1
               offset = pivots[i+1]
              
           except IndexError:
               onset = pivots[i] + 1
               offset = agg.shape[0]
               to_drop = [ x for x in range(onset,offset)]
               indexes_to_drop += to_drop
         
           
           
       print("indexes_to_drop", indexes_to_drop)
       
       agg.drop(df.index[indexes_to_drop], inplace=True)
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
    print("df.empty", df.empty)
    df['date'] = pd.to_datetime(df['date'])
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
   
    if ((min_date == 1) or (max_date == 1)):
        #df.reset_index(level=0, inplace=True)
        return df
    else:
        old_df = df.copy()
        df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
        if df.empty:
            print("df is empty. return old df")
            print("old_df.empty", old_df.empty)
            return old_df
        print("df head", df.head())
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
         except KeyError:
             pass
     return res
    
def csv_to_chunks(sensorId, path, start=None, end=None, n_limit=None):
    print("csv_to_chunks")
    list_df = list()
    df = pd.DataFrame()
    
    if n_limit != None:
        df = pd.read_csv(path, nrows = n_limit)
        
    else:
        df = pd.read_csv(path)
        
    if start != None and end != None:
        df = select_data(df, start, end)
  
    df = downsample(df, '15min')
       
    indexes = find_gap(df, 15)
    list_df = np.split(df, indexes)
    print("list df", list_df)
    
                       
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
 

   
def generate_normal(idSensor, start=None, end=None, simulated=False, n_limit=None, df_to_csv = False, to_chunks=True):
    df = pd.DataFrame()
    if df_to_csv: 
        init_path = os.path.dirname(os.getcwd())
        path = ""
        if simulated == False:
            if init_path == "/content/drive/My Drive/Tese/exploratory":
                path =  init_path + "/wisdom/dataset/infraquinta/real/normal/sensor_"+ str(idSensor) + ".csv"
            else:
                path = init_path + "\\dataset\\infraquinta\\real\\normal\\sensor_" + str(idSensor) + ".csv"
        else:
             if init_path == "/content/drive/My Drive/Tese/exploratory":
                print("drive path simulated")
              
             else:
                path =  init_path + "\\dataset\\simulated\\telegestao\\winter\\sensor_"+ str(idSensor) + ".csv"
                 
        #df = csv_to_df(idSensor, path, limit=limit)
        if to_chunks == True:
            df = csv_to_chunks(idSensor, path, start=start, end=end, n_limit=n_limit)
        else:
            df = csv_to_df(idSensor, path, n_limit=n_limit)
          
    else:
        db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
        conn = create_engine(db_connection)
        query = ""
    
        if n_limit == None:
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
     df['date'] = pd.to_datetime(df['date'])
     aux = df['date'].copy()
     df.index = aux
     df = df.resample(minutes).mean()
     df['date'] = df.index.copy()
     df.index = pd.RangeIndex(start=0, stop=df.shape[0])
     return df
def get_total_max_date(table):
     db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
     conn = create_engine(db_connection)
     query = """SELECT MAX(date)
                FROM %s""" % (table)
     df = pd.read_sql(query, conn)
     res = pd.to_datetime(df.iloc[0,0])
   
     frmt ='%Y-%m-%d %H:%M:%S'
     res = datetime.datetime.strptime(str(res), frmt)
       
     new_frmt = '%d-%m-%Y %H:%M:%S'
     res = datetime.datetime.strftime(res, new_frmt)
  
     return res
     
     
def get_total_min_date(table):
     db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
     conn = create_engine(db_connection)
     query = """SELECT MIN(date)
                FROM %s""" % (table)
                
     df = pd.read_sql(query, conn)
     res = pd.to_datetime(df.iloc[0,0])
   
     frmt ='%Y-%m-%d %H:%M:%S'
     res = datetime.datetime.strptime(str(res), frmt)
       
     new_frmt = '%d-%m-%Y %H:%M:%S'
     res = datetime.datetime.strftime(res, new_frmt)
  
     return res


def generate_total_sequence(idSensor, table, start, end, n_limit=None):
    df = pd.DataFrame()

    init_path = os.path.dirname(os.getcwd())
    path = ""
    if init_path == "/content/drive/My Drive/Tese/exploratory":
        path =  init_path + "/wisdom/dataset/infraquinta/real/sensor_"+ str(idSensor) + ".csv"
    else:
        path = init_path + "\\dataset\\infraquinta\\real\\sensor_" + str(idSensor) + ".csv"
    
    #df = csv_to_df(idSensor, path, limit=limit)
   
    df = csv_to_chunks(idSensor, path, start=start, end=end, n_limit=n_limit)
 
    return df

    return True
def generate_sequences(sensorId, table, start=None, end=None, simulated=False, n_limit = None, df_to_csv = False):
    print("generate_sequences")
    normal_sequence = pd.DataFrame()
    test_sequence = pd.DataFrame()
    total_sequence = generate_total_sequence(sensorId, table, start, end, n_limit=n_limit)
    
    stime = "02-12-2017 00:00:00"
    etime = "31-12-2017 00:00:00"
        
    test_sequence = select_data(total_sequence[0], stime, etime)
    if start!=None and end!=None:
        frmt = '%d-%m-%Y %H:%M:%S'
        start = datetime.datetime.strptime(start, frmt)
        end = datetime.datetime.strptime(end, frmt)
        
    if df_to_csv == True:
        init_path = os.path.dirname(os.getcwd())
        path = ""
        if init_path == "/content/drive/My Drive/Tese/exploratory":
            path =  init_path + "/wisdom/dataset/infraquinta/real/sensor_"+ str(sensorId) + ".csv"
        else:
            path = init_path + "\\dataset\\infraquinta\\real\\sensor_" + str(sensorId) + ".csv"
               
        normal_sequence = generate_normal(sensorId, start=start, end=end, simulated=simulated, n_limit=n_limit, df_to_csv=df_to_csv)
    
    
    else:
        normal_sequence = generate_normal(sensorId, start=start, end=end,simulated=simulated, n_limit=n_limit, df_to_csv = df_to_csv)
       
    print(">>>size normal sequence:", len(normal_sequence))
   
    return normal_sequence, test_sequence