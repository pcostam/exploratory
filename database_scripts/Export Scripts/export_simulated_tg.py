# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:39:47 2020

@author: anama
"""
import pandas as pd
import datetime

def timerange(stime, n):
    actual = 1
    nowtime = stime
    res = list()
    res.append(nowtime.strftime('%Y-%m-%d %H:%M:%S'))
    while actual < n:
        nowtime += datetime.timedelta(minutes=1)  
        frmttime = nowtime.strftime('%Y-%m-%d %H:%M:%S')
        res.append(frmttime)
        actual += 1
    return res

path_init = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\"

files = ['link_flow_summer', 'link_flow_winter', 'node_pressure_summer', 'node_pressure_winter']
epanet_to_scada_pressure = [('aMI817150114', 3), ('aMI817150114',7), 
                             ('aMC402150114',8), ('aMC404150114',11),
                             ('aMC401150114',13),('aMC403150114',15)]


epanet_to_scada_flow = [('6', 1), ('aTU4981150302',10), ('aTU1096150205', 2)
                        , ('aTU455150205',9), ('aTU1477150205',14),('2', 12) ,
                        ('aTU1093150205',6)]  


for file in files:
    print("file:", file)
    map_id = list()
    if 'pressure' in file:
        map_id = epanet_to_scada_pressure
    elif 'flow' in file: 
        map_id = epanet_to_scada_flow
  
  
    path = path_init + "\\simulated\\" + file + ".csv"
    df = pd.read_csv(path)
   
    epanet_names = [i[0] for i in map_id]
    new_df = df[epanet_names]
    df_list = list()
    
    for name in epanet_names:
        series = new_df[name]
        df_list.append(pd.DataFrame(series))
        
    for df in df_list:
        for t in map_id:
            if t[0] in df.columns:
                sensor_id = t[1] 
                df.rename(columns={t[0]: 'value'}, inplace=True)
                
                print("rows", df.shape[0])
                times = timerange(datetime.datetime(2017, 1, 1, 0), df.shape[0])
                print("times", len(times))
                df['date'] = times
                
                path_tmp = ""
                if "winter" in file:
                    path_tmp = path_init + "\\simulated\\telegestao\\winter\\" + "sensor_" + str(sensor_id) + ".csv"
                elif "summer" in file:
                    path_tmp = path_init + "\\simulated\\telegestao\\summer\\"+ "sensor_" + str(sensor_id) + ".csv"
              
                df.to_csv(index=False, path_or_buf=path_tmp)
        
            
    
    