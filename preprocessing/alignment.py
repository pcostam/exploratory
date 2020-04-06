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
    df['value'][(df['value'] < 0) | (df['value'] == 0)] = np.nan
    df = df.interpolate(method='slinear')
    df = df.interpolate(method='nearest').ffill().bfill()
    return df

def alignment(df):    
    return df.resample('1min').interpolate(method='time')


def test():
    df = create_dataset("sensortgmeasure", "1", limit=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(['sensortgId', 'id'], axis = 1)
    df = df.set_index('date')
    df = alignment(df)
    df = missing_values(df)
    print(df)
 
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