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
from base.database import generate_csv
from os import listdir
from os.path import isfile, join
import sys
#sys.path.append('../setup')
from preprocessing.setup import configuration

#https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/


def shift_column(new_df, i, names_columns):
    for name in names_columns:
         new_df[name] = new_df[name].shift(i)
  
    return new_df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, stride=None, dates=False, leaks=False):
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
   df = pd.DataFrame(data)
   
   time = None
   if 'date' in df.columns:
       time = 'date'
   elif 'time' in df.columns:
       time =  'time'
   if time != None:
       df = df.drop([time], axis=1)
       
   n_vars = df.shape[1]
   times_column = list()
   if dates and time != None:
       times_column = data[time]
   del data
   
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
   if dates and time!=None:
       agg[time] = times_column
       
   # drop rows with NaN values  
   if dropnan:
       agg.dropna(inplace=True)
   print("agg", agg)
   return agg
	
  
    
def preprocess_data(df, granularity, start_date, end_date):
    minutes = str(granularity) + "min"
    df = select_data(df, start_date, end_date)
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
    #plot(df['value'][:14])
    #date = df['date'].tolist()
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
    
    df = df_all.drop(['date','sensortgId', 'id'], axis = 1)
    yt = df['value'].tolist()
    decide_input(yt,16,1)
    data = series_to_supervised(df, timesteps)
    data["date"] = df_all["date"]
    X = np.array(data.iloc[:,:timesteps])
    y = np.array(data.iloc[:,timesteps])
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
            return old_df
       
        df.index = pd.RangeIndex(start=0, stop=df.shape[0])
    
        return df
    
def csv_to_df(sensorId, path, limit = True, n_limit=1000):
    
    if limit == True:
        df = pd.read_csv(path, nrows = n_limit)
    else:
        df = pd.read_csv(path)
    if 'value' in df.columns:
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
    #list_df = list()
    df = pd.DataFrame()
    
    if n_limit != None:
        df = pd.read_csv(path, nrows = n_limit)
        
    else:
        df = pd.read_csv(path)
        
    if start != None and end != None:
        df = select_data(df, start, end)
  
    df = downsample(df, '15min')
       
    #indexes = find_gap(df, 15)
    #list_df = np.split(df, indexes)
    #print("list df", list_df)
    
                       
    return df

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
 

def generate_normal_simulation(df, n_train):
    df = df[(df['leak'] == 0)]
    df = df.iloc[:n_train, :]
    print("df shape", df.shape)
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
            df = csv_to_chunks(idSensor, path, start=start, end=end, n_limit=n_limit)
   
        else:
             """
             if init_path == "/content/drive/My Drive/Tese/exploratory":
                print("drive path simulated")
              
             else:
                path =  init_path + "\\dataset\\simulated\\telegestao\\winter\\sensor_"+ str(idSensor) + ".csv"
             """
            
             new_df = preprocess_simulation(idSensor, 18000)
             
             df = generate_normal_simulation(new_df, 10000)
  
          
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

def preprocess_simulation(idSensor, no_leaks):
    from os.path import dirname
    conf_file = join(dirname(__file__), 'setup', 'config.xml')
    root = configuration.read_config(conf_file)
    path = configuration.get_path_simulation(root, simulation="leaks", type_leak="fugas_Q", season=None)
             
    columns = list()
    time_passed = 0
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for file in files[:no_leaks]:
            new_path = os.path.join(root, file)
            df = pd.read_csv(new_path)
            df = df[[idSensor,'leak']]
            no_rows = df.shape[0]
            total_seconds = no_rows*600
            times = [x for x in range(time_passed,time_passed + total_seconds,600)]
            time_passed = time_passed + total_seconds
            df['time'] = times
            df = df.rename(columns={idSensor: 'value'})
            columns.append(df)
            
    agg = pd.concat(columns)
          
    return agg

def generate_total_sequence(idSensor, table, start, end, simulated=False, n_limit=None):
    """Generates the time series (with anomalies)"""
    df = pd.DataFrame()

    init_path = os.path.dirname(os.getcwd())
    path = ""
    if simulated:
        from os.path import dirname

        conf_file = join(dirname(__file__), 'setup', 'config.xml')
        root = configuration.read_config(conf_file)
        path = configuration.get_path_simulation(root, simulation="leaks", type_leak="fugas_Q", season=None)
        df = preprocess_simulation(path, idSensor, 10000)
    else:
        if init_path == "/content/drive/My Drive/Tese/exploratory":
            path =  init_path + "/wisdom/dataset/infraquinta/real/sensor_"+ str(idSensor) + ".csv"
        else:
            path = init_path + "\\dataset\\infraquinta\\real\\sensor_" + str(idSensor) + ".csv"
        
        df = csv_to_chunks(idSensor, path, start=start, end=end, n_limit=n_limit)
 
    return df 

    return True

def change_format(res, frmt, new_frmt):
    frmt ='%Y-%m-%d %H:%M:%S'
    res = datetime.datetime.strptime(str(res), frmt)
       
    new_frmt = '%d-%m-%Y %H:%M:%S'
    res = datetime.datetime.strftime(res, new_frmt)
    
    return res
def generate_sequences(sensorId, table, start=None, end=None, simulated=False, n_limit = None, df_to_csv = False):
    normal_sequence = pd.DataFrame()
    test_sequence = pd.DataFrame()
    total_sequence = generate_total_sequence(sensorId, table, start, end, n_limit=n_limit)
    size_total_sequence = total_sequence.shape[0]
    size_train = round(size_total_sequence*0.8)
    sequence_train = total_sequence.iloc[:size_train, :]

    if start == None and end == None:
        start = str(min(sequence_train['date']))
        end = str(max(sequence_train['date']))
    
    test_sequence = total_sequence.iloc[size_train:, :]
    print("start", start)
    print("end", end)
    
    #test_sequence = select_data(total_sequence[0], stime, etime)
    if start!=None and end!=None:
        start = change_format(start, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
        end = change_format(end, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
        
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
       
    return normal_sequence, test_sequence



def split_train_test(normal_sequence, all_sequence, n_train, test_split=0.2, gap=0, time="date"):
    """
    Splits sequence into normal sequence(to train) and test sequence
    (normal and anomalous), making into chunks.
    
    Parameters
    ----------
    Dataframe
    n_train : int
    Size of each chunk.
    
    Returns
    -------
    None.

    """
    n_test = get_no_instances_test(n_train, test_split)
    print("n_train", n_train)
    print("n_test", n_test)
    margin = 0
    train_chunks = list()
    test_chunks = list()
    n_records = len(all_sequence)
    start_date = normal_sequence.iloc[0][time]
    
    while margin < n_records:
        train = normal_sequence[(normal_sequence[time] >= start_date)]
        if train.empty:
            break
        train = train.iloc[:n_train]
        print("train", train)
        #Date that should begin anomalous sequence for test sequence
        start_date = train.iloc[-1][time]
        test = all_sequence[(all_sequence[time] >= start_date)]
        test = test[:n_test]   
        #Date that should end anomalous sequence for test sequence
        #and begin new train chunk
        end_date = test.iloc[-1][time]
        start_date = end_date

        test_chunks.append(test)
        train_chunks.append(train)
        
        margin += n_train - 1
        
    return train_chunks, test_chunks

    
    
    
    

def get_no_instances_test(n_train, test_split=0.2):
    """ 
    Returns number where of necessary instances to make the 
    percentage test split necessary
    Parameters
    ----------
    n_train : int
        number of train instances
    test_split : float, optional
        percentage of test sequence
    
    """
    return round((test_split*n_train)/(1-test_split))
#blocked margin = n_train + n_test
def rolling_out_cv(X, n_train, test_split=0.2, gap=0, blocked=True):
    """ With X, divides in train and test chunks """
    margin = 0
    train_chunks = list()
    test_chunks = list()
    n_records = len(X)
    start = 0
    
    n_test = round((test_split*n_train)/(1-test_split))
    print("n_val", n_test)

    i = 0
    while margin < n_records:
        start = i + margin
        i += 1
        stop = start + n_train
        train = X[start:stop]
        
        #index test set
        start = stop + gap
        stop = start + n_test
        
        if X[start:stop].empty:
            break
        
       
        test_chunks.append(X[start:stop])
        train_chunks.append(train)
        if blocked:
            margin += n_train + n_test -1
        else:
            margin += n_train - 1
        
    return train_chunks, test_chunks
    
  
def test():
    col_1 = [x for x in range(40)]
    col_2 = [x for x in range(5, 35)]
    dt = datetime.datetime(2010, 12, 1)
    end = datetime.datetime(2010, 12, 30, 23, 59, 59)
    step = datetime.timedelta(minutes=15)
    dates_list = []
    while dt < end:
        dates_list.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
        dt += step
        
    dates1 = dates_list[:40]
    dates2 = dates_list[5:35]
    d = {'value': col_2, 'date':dates2}
    d2 = {'value': col_1, 'date': dates1}
    df1 = pd.DataFrame(data=d)
    df2 = pd.DataFrame(data=d2)
    #train_chunks, validation_chunks = rolling_out_cv(df1, n_train=6)
    
    train_chunks, test_chunks = split_train_test(df1, df2, n_train=6, test_split=0.2, gap=0)
    for train,test in zip(train_chunks, test_chunks):
        print("train", train)
        print("test", test)
    
        

   
    
    