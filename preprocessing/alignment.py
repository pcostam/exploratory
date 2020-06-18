# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:29:54 2020

@author: anama
"""
import pandas as pd
import numpy as np 
from preprocessing.series import create_dataset
from sqlalchemy import create_engine

def missing_values(df):
    print("missing values")
    print("before", df.head())
    df['value'][(df['value'] < 0) | (df['value'] == 0)] = np.nan
    print("Number of nan values:",  df.isna().sum())
    print(df.head())
    print(df.shape)
    df = df.interpolate(method='time')
    df = df.resample('1min')
    df = df.interpolate(method='slinear').ffill().bfill()
    print("after", df[:9])
    print("Number of nan values:",  df.isna().sum())
    return df

def alignment(df):   
    print("aligment")
    print("before", df.head())
    print("Number of nan values:",  df.isna().sum())
    print(list(df.columns))
    print(df.columns)
    print("dtypes", df.dtypes)
    df = df[['value']]
    df['value'] = pd.to_numeric(df['value'])
    print("dtypes", df.dtypes)
    print("index", type(df.index))
    print(df.columns)
    df = df.reset_index()
    df = df.set_index(['date'])
    resampled = df.resample('1min')
    df = df.reset_index()
    df = df.set_index(['date'])
    interpolated = resampled.interpolate(method='nearest')
    print("after", interpolated.head())
    print("Number of nan values:",  interpolated.isna().sum())
    # interpolated = interpolated.interpolate(method='nearest').ffill().bfill()
    #print("Number of nan values:",  interpolated.isna().sum())
    return interpolated 
  
    

def test():
    df = create_dataset("sensortgmeasure", "1", limit=True)
    print(df.head())
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = missing_values(df)
    #df = alignment(df)
    #df = missing_values(df)
    print(df.head())
    #
    #print(df.head())
 
#create table sensortgmeasurepp
def populate():
    df_list = list()
    
    for id_tg in range(1, 16):  
        print("id_tg>>>", id_tg)
        df = create_dataset("sensortgmeasure", str(id_tg), limit = False)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop(['sensortgId', 'id'], axis = 1)
        df = df.set_index('date')
        df = alignment(df)
        df = missing_values(df)
        df['sensortgId'] = id_tg
        df = df.reset_index()
        df_list.append(df)
       
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df.sort_values(by=['date'])
  
  
    db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    conn = create_engine(db_connection)
    df.to_sql('sensortgmeasurepp', con=conn, if_exists='append', index=False)
    
    return True