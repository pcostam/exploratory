# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:57:19 2020

@author: anama
"""
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
def make_connection(db_name):
        db = mysql.connector.connect(
                host="127.0.0.1",
                port="3306",
                user="root",
                passwd="banana",
                database= db_name)
        cursor = db.cursor(buffered=True)
        return db, cursor
   

def select_data(min_date, max_date, sensorId):
    #db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    #conn = create_engine(db_connection)
    query = """
    SELECT ATG.date, STM.value, ATG.anomaly
    FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
    WHERE idmeasurestg=STM.id AND ATG.idSensor=%s AND
    ATG.date BETWEEN '%s' AND '%s'
    """ % (sensorId, min_date, max_date)
  

    #df = pd.read_sql(query , conn)
    path = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\infraquinta\\mask\\sensor_"+ str(sensorId) + ".csv"
    generate_csv(query, sensorId, path)
    
    #return df
    
def write_mask(sensorId):
    print("Start write folder mask")
    #db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    #conn = create_engine(db_connection)
    query = """
    SELECT ATG.date, STM.value, ATG.anomaly
    FROM infraquinta.anomaliestg as ATG, infraquinta.sensortgmeasurepp as STM
    WHERE idmeasurestg=STM.id AND ATG.idSensor=%s 
    """ % (sensorId)
  
    path = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\infraquinta\\real\\mask\\sensor_"+ str(sensorId) + ".csv"
    generate_csv(query, sensorId, path)
    print("End write folder mask")
    
    
def write_to_disk(command="mask"):
    if command == "mask":
        for sensorId in range(1,16):
            write_mask(sensorId)
            
            
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

    