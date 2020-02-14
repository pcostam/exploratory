# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:41:22 2020

@author: anama
"""
from matplotlib import pyplot
from preprocessing.series import create_dataset
from sqlalchemy import create_engine, text, update
import numpy as np
import mysql.connector
from sklearn.externals import joblib
from scipy.stats import shapiro, normaltest, anderson
import pandas as pd
import datetime
import base64
from io import BytesIO
from scipy.stats import norm
import Pvalue

from scipy.stats import boxcox, yeojohnson
class Sensor:
    def __init__(self, name, idSensor):
        self.name = name
        self.id = idSensor
        self.instant = -1
 
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
        
        
        
    def update(self, actual_measure, prev_measure, neighbors_ids, neighbors_measures, instant):
        self.instant += 1
        diff = float(round(prev_measure - actual_measure, 2))
        
        db, cursor = Sensor.make_connection("infraquinta")
        Sensor.populate_differences(db, cursor, self.id, diff, instant, self.id)
      
        for i in range(0, len(neighbors_measures)):
                diff = float(round(actual_measure - neighbors_measures[i],2))
                Sensor.populate_differences(db, cursor, self.id, diff, instant, neighbors_ids[i])
        
        cursor.close()
        db.close()
       
    
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
        df = create_dataset("sensortgmeasure", "12", limit=False)
        value = list(df["value"])
        print("value", len(value))
        
        neighbor_1 = create_dataset("sensortgmeasure", "10", limit=False)
        neighbor_1_val = neighbor_1["value"]
        neighbor_2 = create_dataset("sensortgmeasure", "1", limit=False)
        neighbor_2_val = neighbor_2["value"]
      
        sensor = Sensor("RPR Caudal Grv", 12)
        
        for i in range(0,len(df)):
            try:
                sensor.update(value[i], value[i-1], [10,1], [neighbor_1_val[i], neighbor_2_val[i]], str(df['date'][i]))
            except IndexError:
                sensor.update(value[i], 0, [10,1], [neighbor_1_val[i], neighbor_2_val[i]], str(df['date'][i]))
    
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
        html_string = """<html>
        <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
        </head><body>""" + all_encoded[0][0] + """<img src=\'data:image/png;base64,{}\'>'""".format(all_encoded[0][1]) + """<div>"""+ all_encoded[1][0] + """<img src=\'data:image/png;base64,{}\'>'""".format(all_encoded[1][1])+ """</div>""" + all_encoded[2][0] + """<img src=\'data:image/png;base64,{}\'>'""".format(all_encoded[2][1]) + """</div></body></html>"""
        f = open('report.html','w')
        f.write(html_string)
        f.close()
    def test():
        db, cursor = Sensor.make_connection("infraquinta")
        
        query = "SELECT difference FROM differencesmeasuretg WHERE idSensor=%s and diffWith=%s"
        sensorId = 12 
        diffId = 1
        
        cursor.execute(query, (sensorId, diffId))
        print("type", type(cursor))
        
        measures = list(measure[0] for measure in cursor)

        sample = measures
        
        #no transformation
        Sensor.check_for_normality(sample)
        #Fit a normal distribution to the data:
        mu, std = norm.fit(sample)
        Pvalue.pvalue_norm(mu, std)
        
        plot_url = Sensor.plot_diff_measure(sample, "Histogram with differences between sensor 12 and sensor 1 with no power transformation", "sensor-12-1", mu, std)
        
        all_url = [("Histogram with differences between sensor 12 and sensor 1 with no power transformation", plot_url)] 
     
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
    
     
        
        #sample = np.array(sample)
        #sample = sample.reshape((len(sample), 1))
        #density function
        #model = KDE(sample)
        
        #model = joblib.load('KDE.pkl')
        
        #values = np.linspace(-50, 150, 48360)
        #values = values.reshape((len(sample), 1))
        
        #probabilities = model.score_samples(values)
        #probabilities = np.exp(probabilities)
        # print("probabilities", probabilities)
        
        # approx. 10 percent of smallest pdf-values: lets treat them as outliers 
        #outlier_inds = np.where(probabilities < np.percentile(probabilities, 10))[0]
        #outliers = [values[i] for i in outlier_inds]
       
        
        #Sensor.plot_diff_measure(sample, values, probabilities, markers=outliers)
        plot_url = Sensor.plot_diff_measure(posdata, "Histogram with differences between sensor 12 and sensor 1 applying Box-Cox power transformation", "box-cox-sensor-12-1", mu, std)
        all_url.append(("Histogram with differences between sensor 12 and sensor 1 applying Box-Cox power transformation", plot_url))
        
       
        
        
        data, lmbda = yeojohnson(sample)
       
        mu, std = norm.fit(data)
        Pvalue.pvalue_norm(mu, std)
        Sensor.check_for_normality(data)
        plot_url = Sensor.plot_diff_measure(data, "Histogram with differences between sensor 12 and sensor 1 applying Yeo-Johnson power transformation", "yeo-johnson-sensor-12-1", mu, std)
        all_url.append(("Histogram with differences between sensor 12 and sensor 1 applying Yeo-Johnson power transformation", plot_url))
        
        
        
        Sensor.write_report(all_url)
        cursor.close()
        db.close()
        
        
#todo
      
        
    
   