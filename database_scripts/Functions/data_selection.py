# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd

def process_df(df):
    
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df.index = df['date']
    del df['date']
    return df

def select_data(path_init, wme, type, sensor_id, min_date, max_date):
        
    path = path_init + "\\Data\\" + wme + "\\" + type + "\\sensor_" + str(sensor_id) + ".csv"
    df = pd.read_csv(path)
    df = process_df(df) 
    
    if ((min_date == 1) or (max_date == 1)):
        return df
    else:
        df = df[(df.index >= min_date) & (df.index <= max_date)]
        return df