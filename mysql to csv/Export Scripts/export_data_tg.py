# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:48:49 2020

@author: susan

Exports telemanagement (telegestao) data to a csv -> one sensor per file

"""

import mysql.connector
import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *

root = read_config()
path_init = get_path(root)
db_config = get_db(root)
wmes = get_wmes(root)

mydb = mysql.connector.connect(
  host=db_config['host'],
  user=db_config['user'],
  passwd=db_config['pw']
)

print(mydb)
print("\nExport initiated")

cursor = mydb.cursor(buffered=True)

for wme in wmes:
    path = path_init + "\\Data\\" + wme + "\\real\\sensor_"
    query = "SELECT count(*) FROM " + wme + ".sensortg"
      
    cursor.execute(query)
    sensor_count = cursor.fetchall()[0][0]
    
    print("\n" + wme + ": " + str(sensor_count) + " sensors found")
    
    sensor_id = 1
    
    for i in range(1,sensor_count+1):
            
        query = "SELECT date, value FROM " + wme + ".sensortgmeasure where sensortgId = " + str(sensor_id)
        path_tmp = path + str(sensor_id) + ".csv"
        
        df = pd.read_sql(query, con=mydb)
        df.to_csv(index=False, path_or_buf=path_tmp)
        
        print("  sensor " + str(sensor_id) + ": " + str(df.shape[0]) + " rows")
        
        sensor_id += 1
        
print("\nExport completed")

cursor.close()
mydb.close()