# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:41:22 2020

@author: anama
"""
from matplotlib import pyplot
from preprocessing.series import create_dataset
from sqlalchemy import create_engine, text
import numpy as np
import mysql.connector
#from sklearn.externals import joblib
from scipy.stats import shapiro, normaltest, anderson
import pandas as pd
import datetime
import base64
from io import BytesIO
from scipy.stats import norm
from Pvalue import Pvalue
from scipy.stats import boxcox, yeojohnson

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'F:\\Tese\\exploratory\\wisdom\\report')
from image import Image

class Sensor(object):
    flowSensorsIds = [1, 2, 4, 6, 9, 10, 12, 14]
    pressureSensorsIds = [3, 5, 7, 8, 11, 13, 15]
    #in order
    sensorNeighbors =  {1:[10,14,12], 2:[4,6,9,12], 3:[7,5,8,11], 4:[6,2,9,12], 5:[7,3,9,11], 
                                 6:[4,2,9,12], 7:[5,3,8,11], 8:[11,13,5,7], 9:[12, 14, 4, 6], 10:[1, 14, 12], 
                                 11:[13, 8, 15, 5, 7], 12:[14, 9, 4, 6], 13:[15, 11, 8], 14:[10, 12], 15:[13,11,8]}
    @classmethod
    def getSensorNeighbors(cls):
        return cls.sensorNeighbors
    
    @classmethod
    def getFlowSensorsIds(cls):
        return cls.flowSensorsIds
    
    @classmethod
    def getPressureSensorsIds(cls):
        return cls.pressureSensorsIds
    
    @classmethod
    def getSensorIds(cls):
        return cls.sensorNeighbors.keys()
    
    def make_connection(db_name):
        db = mysql.connector.connect(
                host="127.0.0.1",
                port="3306",
                user="root",
                passwd="banana",
                database= db_name)
        cursor = db.cursor()
        return db, cursor
    
        
    def populate_differences(db, cursor, idSensor, diff, instant, diffWith):
    
        query = "INSERT into differencesmeasuretg (idSensor, difference, instant, diffWith) VALUES (%s, %s, %s, %s)"
        
        cursor.execute(query, (idSensor, diff, instant, diffWith))
        db.commit()
        
        
        
    def update(idSensor, db, cursor, actual_measure, prev_measure, neighbors_ids, neighbors_measures, instant):
        diff = float(round(prev_measure - actual_measure, 2))
        
        Sensor.populate_differences(db, cursor, idSensor, diff, instant, idSensor)
      
        size = len(neighbors_measures)
        for i in range(0, size):
                diff = float(round(neighbors_measures[i]-actual_measure ,2))
                Sensor.populate_differences(db, cursor, idSensor, diff, instant, neighbors_ids[i])
        
     
    
    def plot_diff_measure(sample, tile, filename, mu, std, values=[], probabilities=[], markers=[]):
        #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
        fig = pyplot.figure()
        pyplot.hist(sample, bins=50, density=True)
      
        
        
        if probabilities != [] and values != []:
            if markers == [] :
                pyplot.plot(values, probabilities)
            else:
                pyplot.plot(values, probabilities, markevery=markers) 
                
         #plot pdf
        xmin, xmax = pyplot.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        pyplot.plot(x, p, 'k', linewidth=2)
        
        
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        
        pyplot.show()
        
       
        return encoded
        
    def write_to_disk():
        db, cursor = Sensor.make_connection("infraquinta")
        for idSensor in Sensor.getSensorIds():
            df = create_dataset("sensortgmeasure", str(idSensor), limit=False)
            value = list(df["value"])
            neighbors = []
            idNeighborsList = list(Sensor.getSensorNeighbors()[idSensor])
            for idNeighbor in idNeighborsList:
                df_neighbor = create_dataset("sensortgmeasure", str(idNeighbor), limit=False)
                df_neighbor_val = df_neighbor["value"]
                neighbors.append(df_neighbor_val)
            
            print("idsensor", idSensor)
            print(neighbors)
         
            size = len(value)
         
            for i in range(0,size):
                try:
                    Sensor.update(idSensor, db, cursor, value[i], value[i-1], idNeighborsList, [neighbor[i] for neighbor in neighbors], str(df['date'][i]))
                except IndexError:
                    Sensor.update(idSensor, db, cursor, value[i], 0, idNeighborsList, [neighbor[i] for neighbor in neighbors], str(df['date'][i]))
                except KeyError as e:
                    print("ERROR", e)
        
        Sensor.close_connection(cursor, db)
            
    def close_connection(cursor, db):
        cursor.close()
        db.close()
        
    def write_to_disk_anomalies(idSensor):
         db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
         conn = create_engine(db_connection)
         query = "SELECT * FROM ordemdata"
         df = pd.read_sql(query , conn)
         
         query = "SELECT * FROM sensortgmeasure where sensortgId="+ idSensor
         dates_measures = pd.read_sql(query , conn)['date']
         
         for date in dates_measures:
              query = text("""INSERT INTO anomaliestg (idSensor, date, anomaly) VALUES(:id, :date, :anomaly)""")    
              conn.execute(query, (idSensor, date, 0))
              
         for index, row in df.iterrows(): 
             if row['descricao']=='Percepcao':
                 date_event = row['date']
                
                 #find dates 2 days before and consider it as anomaly
                 date_N = Sensor.date_N_days_ago(date_event, 2)
                 for date in dates_measures:
                     if date >= date_N and date <= date_event:
                          #update 
                          query = text("""UPDATE anomaliestg SET anomaly = :anomaly WHERE date= :date""")
                          conn.execute(query, (1, date))
                         
    def date_N_days_ago(date, days):
        return date - datetime.timedelta(days=days)     
    
    def check_for_normality(sample):
        # normality test Shapiro-Wilk test
        stat, p = shapiro(sample)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
            
        #normality test d'agostino        
        stat, p = normaltest(sample)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
            
        #anderson-darling test
        result = anderson(sample)
        print('Statistic: %.3f' % result.statistic)
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    
    def write_report(all_encoded):
        html_string = """
        <html>
        <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
        </head><body>""" 
        
        for img in all_encoded:
            html_string += Image.to_html(img)
        
        end_string ="""
        </body></html>
        """
        
        html_string += end_string
        
        f = open('report.html','w')
        f.write(html_string)
        f.close()
    
  
        
    def pvalue_analysis(sample, sensorId, diffId, cursor, db):
        #no transformation
        Sensor.check_for_normality(sample)
        #Fit a normal distribution to the data:
        mu, std = norm.fit(sample)
        Pvalue.pvalue_norm(mu, std)
        
        plot_url = Sensor.plot_diff_measure(sample, "Histogram with differences between sensor 12 and sensor 1 with no power transformation", "sensor-12-1", mu, std)
        
        all_url = [("Histogram with differences between sensor %s and sensor %s with no power transformation" % (sensorId, diffId), plot_url)] 
        
        all_imgs = []
        """
        #box-cox transformation
        print("min:" , min(sample))
        
        shift = 0
        minimum = min(sample)
        if minimum < 0:
            shift = round(abs(minimum))
     
        posdata = [x + shift for x in sample] 
        
        posdata, lmda = boxcox(posdata)
      
        mu, std = norm.fit(posdata)
        Pvalue.pvalue_norm(mu, std)
    

        plot_url = Sensor.plot_diff_measure(posdata, "Histogram with differences between sensor %s and sensor %s applying Box-Cox power transformation" % (sensorId, diffId), "box-cox-sensor-12-1", mu, std)
        all_url.append(("Histogram with differences between sensor %s and sensor %s applying Box-Cox power transformation" % (sensorId, diffId), plot_url))
        """
        data, lmbda = yeojohnson(sample)
       
        mu, std = norm.fit(data)
        crits, pvaluesList = Pvalue.pvalue_norm(mu, std)
        
        Sensor.check_for_normality(data)
        plot_url = Sensor.plot_diff_measure(data, "Histogram with differences between sensor %s and sensor %s applying Yeo-Johnson power transformation" % (sensorId, diffId), "yeo-johnson-sensor-12-1", mu, std)
        
        img = Image("Histogram with differences between sensor %s and sensor %s applying Yeo-Johnson power transformation" % (sensorId, diffId), plot_url)
        all_imgs.append(img)
        
        all_url.append(("Histogram with differences between sensor %s and sensor %s applying Yeo-Johnson power transformation" % (sensorId, diffId), plot_url))
        
        Sensor.write_report(all_imgs)
        
        return pvaluesList[0], crits[0]
        
    def test(sensorId=None):
        db, cursor = Sensor.make_connection("infraquinta")
        critsList = []
        pvaluesList = []
        if sensorId==None:
            for sensorId in Sensor.getSensorIds():
                for diffId in Sensor.getSensorNeighbors()[sensorId]:
                    query = "SELECT difference FROM differencesmeasuretg WHERE idSensor=%s and diffWith=%s"
                
                    cursor.execute(query, (sensorId, diffId))
                    print("type", type(cursor))
                
                    measures = list(measure[0] for measure in cursor)
                    crits, pvaluesList = Sensor.pvalue_analysis(measures, sensorId, diffId, cursor, db)
        else:
           
            ids_compare = Sensor.getSensorNeighbors()[sensorId] + [sensorId]
            for diffId in ids_compare:
                query = "SELECT difference FROM differencesmeasuretg WHERE idSensor=%s and diffWith=%s"
                
                cursor.execute(query, (sensorId, diffId))
                print("type", type(cursor))
                
                measures = list(measure[0] for measure in cursor)
                pvalues, crits = Sensor.pvalue_analysis(measures, sensorId, diffId, cursor, db)
             
                critsList.append(crits)
                pvaluesList.append(pvalues)
            
            Pvalue.consensus(pvaluesList, critsList)
            
        cursor.close()
        db.close()
        
            

       
        
        
#todo
      
        
    
   