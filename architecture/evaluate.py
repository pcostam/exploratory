# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:17:04 2019

@author: anama
"""

from preprocessing.series import create_dataset
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from matplotlib import pyplot 
from pandas.plotting import register_matplotlib_converters



def evaluate():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    f = open('out.txt', 'w')
    register_matplotlib_converters()
    df = create_dataset("sensortgmeasure", "12", limit = False)
    
    #df = df.set_index(pd.DatetimeIndex(df['date']))
    df['value'][df['value'] < 0] = np.nan
    df = df.interpolate(method='slinear')
    df.interpolate(method='nearest').ffill().bfill()
    flow12 = df['value']
    f.write("flow 12 describe" + str(flow12.describe()))
    
 


    df_flow = create_dataset("sensortgmeasure", "14", limit=False)
    df_flow['value'][df_flow['value'] < 0] = np.nan
    #https://stackoverflow.com/questions/56941316/interpolation-still-leaving-nans-pandas-groupby
    df_flow = df_flow.interpolate(method='slinear')
    df.interpolate(method='nearest').ffill().bfill()
    flows = df_flow['value']
    print("flows 14", flows)
    
    f.write("flow 14 describe" + str(flows.describe()))
    df_concat = pd.DataFrame()

  
    df_concat["flow12"] = flow12
    df_concat["flow14"] = flows
    df_concat["flow1"] = create_dataset("sensortgmeasure", "1", limit=False)['value']
    df_concat["flow2"] = create_dataset("sensortgmeasure", "2", limit=False)['value']
    df_concat["flow4"] = create_dataset("sensortgmeasure", "4", limit=False)['value']
    df_concat["flow6"] = create_dataset("sensortgmeasure", "6", limit=False)['value']
    df_concat["flow9"] = create_dataset("sensortgmeasure", "9", limit=False)['value']
    df_concat["flow10"]= create_dataset("sensortgmeasure", "10", limit=False)['value']
          
    f.write("covariance\n"+ str(df_concat.cov()))
    f.write("correlations" + str(df_concat.corr()))
    
    
    
    
    df_pressure14 = create_dataset("sensortgmeasure", "13", limit=False) 
    df_pressure14['value'][df_pressure14['value'] < 0] = np.nan
    df_pressure14 = df_pressure14.interpolate(method='slinear')
    df_pressure14.interpolate(method='nearest').ffill().bfill()
    f.write("describe pressure 14" + str(df_pressure14['value'].describe()))
    print(">>>sensor 13")
    pressure14 = df_pressure14['value']
    print("pressure14", pressure14)
    
    #get_pressure_drop_stats(pressure14)
    
    df_pressure12 = create_dataset("sensortgmeasure", "11", limit=False) 
    df_pressure12['value'][df_pressure12['value'] < 0] = np.nan
    df_pressure12 = df_pressure12.interpolate(method='slinear')
    df_pressure12.interpolate(method='nearest').ffill().bfill()
    f.write("describe pressure 12\n" + str(df_pressure12['value'].describe()))
    print(">>>sensor 11")
    #pressure12 = df_pressure12['value']
    #get_pressure_drop_stats(pressure12)
    
    
    df_flow_pressure = pd.DataFrame()
    df_flow_pressure["flow12"] = flow12
    df_flow_pressure["flow14"] = flows
    df_flow_pressure["pressure12"] = df_pressure12['value']
    df_flow_pressure["pressure14"] = df_pressure14['value']
    f.write("correlations pressure and flow" + str(df_flow_pressure.corr()))
    
    f.close()
    
                
def get_pressure_drop_stats(pressures):
    previous = next_ = None
    l = len(pressures)
    pressure_drop = False
    begin_pressure = 0
    end_pressure = 0
    all_drops = list()
    for index, obj in enumerate(pressures):
            if index > 0:
                previous = pressures[index - 1]
            if index < (l - 1):
                next_ = pressures[index + 1]
                
            if previous and next_ != None:
                if previous - next_ > 0:
                    if pressure_drop == False:
                        begin_pressure = previous
                    pressure_drop = True
                if previous - next_ < 0:
                    if pressure_drop == True:
                        end_pressure = previous
                        drop = begin_pressure - end_pressure
                        all_drops.append(drop)
                    pressure_drop = False

                
    all_drops = np.array(all_drops)
    print("mean pressure drop", np.mean(all_drops))
    print("max pressure drop", np.max(all_drops))
    
    
def f_beta_score(TP, FP, FN, beta=0.5):
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    
    return (1+beta**2)*((precision*recall)/((beta**2)*precision)+recall)

    