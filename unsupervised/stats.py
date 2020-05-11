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
from unsupervised.Pvalue import Pvalue
from unsupervised.Sensor import Sensor
from scipy.stats import boxcox, yeojohnson
import time
from report import image, HtmlFile, tag, Text
from base.Network import Network
 
class Stats(object):
    flowSensorsIds = [1, 2, 4, 6, 9, 10, 12, 14]
    pressureSensorsIds = [3, 5, 7, 8, 11, 13, 15]
    #in order
    sensorNeighbors =  {1:[10,14,12], 2:[4,6,9,12], 3:[7,5,8,11], 4:[6,2,9,12], 5:[7,3,9,11], 
                                 6:[4,2,9,12], 7:[5,3,8,11], 8:[11,13,5,7], 9:[12, 14, 4, 6], 10:[1, 14, 12], 
                                 11:[13, 8, 15, 5, 7], 12:[14, 9, 4, 6], 13:[15, 11, 8], 14:[10, 12], 15:[13,11,8]}
    
    file = HtmlFile.HtmlFile()
    
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
        cursor = db.cursor(buffered=True)
        return db, cursor
    
        
    def populate_differences(db, cursor, idSensor, diff, instant, diffWith):
    
        query = "INSERT into differencesmeasuretg (idSensor, difference, instant, diffWith) VALUES (%s, %s, %s, %s)"
        
        cursor.execute(query, (idSensor, diff, instant, diffWith))
        db.commit()
        
        
        
    def update(idSensor, db, cursor, actual_measure, prev_measure, neighbors_ids, neighbors_measures, instant):
        diff = float(round(prev_measure - actual_measure, 2))
        
        Stats.populate_differences(db, cursor, idSensor, diff, instant, idSensor)
      
        size = len(neighbors_measures)
        for i in range(0, size):
                diff = float(round(neighbors_measures[i]-actual_measure ,2))
                Stats.populate_differences(db, cursor, idSensor, diff, instant, neighbors_ids[i])
        
     
    
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
        
    def write_to_disk_differences(db, cursor, idSensor):
        df = create_dataset("sensortgmeasure", str(idSensor), limit=False)
        value = list(df["value"])
        neighbors = []
        idNeighborsList = list(Stats.getSensorNeighbors()[idSensor])
        for idNeighbor in idNeighborsList:
            df_neighbor = create_dataset("sensortgmeasure", str(idNeighbor), limit=False)
            df_neighbor_val = df_neighbor["value"]
            neighbors.append(df_neighbor_val)
            
            print("idsensor", idSensor)
            print(neighbors)
         
            size = len(value)
         
        for i in range(0,size):
                try:
                    Stats.update(idSensor, db, cursor, value[i], value[i-1], idNeighborsList, [neighbor[i] for neighbor in neighbors], str(df['date'][i]))
                except IndexError:
                    Stats.update(idSensor, db, cursor, value[i], 0, idNeighborsList, [neighbor[i] for neighbor in neighbors], str(df['date'][i]))
                except KeyError as e:
                    print("ERROR", e)
            
            
   
    def close_connection(cursor, db):
        cursor.close()
        db.close()
        
    def write_to_disk_anomalies():
         db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
         conn = create_engine(db_connection)
         query = "SELECT * FROM ordemdata"
        
         query = "SELECT * FROM sensortgmeasurepp"

         measures = pd.read_sql(query , conn)
         
         for index, row in measures.iterrows():
              query = text("""INSERT INTO anomaliestg (idSensor, date, anomaly, idmeasurestg) VALUES(:idSensor, :date, :anomaly, :idmeasurestg)""")    
              conn.execute(query, idSensor=row["sensortgId"], date=str(row["date"]), anomaly=0, idmeasurestg=row["id"])
              
         Stats.update_interval_anomalies()
                          
    def update_interval_anomalies():
        start_time = time.time()
        print("start_time")
        db, cursor = Stats.make_connection("infraquinta")
        query = "SELECT date FROM ordemdata WHERE descricao='Percepcao'"
                    
        cursor.execute(query)
      
        dates_start = list(date[0] for date in cursor)
        ids = list()
        print("size", len(dates_start))
        
        query = "SELECT date FROM ordemdata WHERE descricao='Abertura'"
        cursor.execute(query)
        dates_abertura = list(date[0] for date in cursor)
        dates_end_event = list()
        
        for abertura in dates_abertura:
            for fecho in dates_start:
                 diff = abertura - fecho
                 print("diff", diff)
                 diff_hours = diff.total_seconds() /3600
                 print("diff hours", diff_hours)
                 if diff_hours < 16 and diff_hours > 0:
                     dates_end_event.append(abertura)
                 
        print("len abertura", len(dates_end_event))
                 
            
        #select opening that are after a percepcao
        for date_event, date_perception in zip(dates_end_event, dates_start):
            date_N = Stats.date_N_days_ago(date_perception, 2)
            print("date_N", date_N)
            print("date_event", date_event)
            query="""
            SELECT id
            FROM infraquinta.anomaliestg as ATG
            WHERE (date > %s AND date < %s)
            """     
            cursor.execute(query, (date_N,date_event))
            ids = list(idSensor[0] for idSensor in cursor)
            
        for idSensor in ids:
            query = "UPDATE anomaliestg SET anomaly=1 WHERE id=%s "
            cursor.execute(query, (idSensor,))
            db.commit()
            
        Stats.close_connection(cursor, db)
        elapsed_time = time.time() - start_time
        print("elapsed_time" + str(elapsed_time) + "seconds")
       
        
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
    
    def write_report(file_name):
        Stats.file.writeToHtml(file_name)
        
    def init_report(file_name, all_encoded):
        Stats.file = HtmlFile.HtmlFile()
        html = tag.Html()
        Stats.file.append(html)
        head = tag.Head()
        Stats.file.append(head)
        body = tag.Body()
       
        for img in all_encoded:
            #html_string += Image.to_html(img)
            body.append(img)
        Stats.file.append(body)
        
        
     
    
  
        
        
    def pvalue_analysis(sample, sensorId, diffId, cursor, db):
        all_imgs = []
        #no transformation
        Stats.check_for_normality(sample)
        #Fit a normal distribution to the data:
        mu, std = norm.fit(sample)
        Pvalue.pvalue_norm(mu, std)
        
        plot_url = Stats.plot_diff_measure(sample, "Histogram with differences between sensor 12 and sensor 1 with no power transformation", "sensor-12-1", mu, std)        
        img = image.Image("Histogram with differences between sensor %s and sensor %s with no powe transformation" % (sensorId, diffId), plot_url)
        all_imgs.append(img)
        crit, pvalue = Pvalue.pvalue_norm(mu, std, percentage=0.05)
     
        """"
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
    

        plot_url = Stats.plot_diff_measure(posdata, "Histogram with differences between sensor %s and sensor %s applying Box-Cox power transformation" % (sensorId, diffId), "box-cox-sensor-12-1", mu, std)
        all_url.append(("Histogram with differences between sensor %s and sensor %s applying Box-Cox power transformation" % (sensorId, diffId), plot_url))
        """
        """
        data, lmbda = yeojohnson(sample)
        print("DATA YEO>>", data)
        
       
        mu, std = norm.fit(data)
        crit, pvalue = Pvalue.pvalue_norm(mu, std, percentage=0.02)
        
        Stats.check_for_normality(data)
        plot_url = Stats.plot_diff_measure(data, "Histogram with differences between sensor %s and sensor %s applying Yeo-Johnson power transformation" % (sensorId, diffId), "yeo-johnson-sensor-12-1", mu, std)
        
        img = image.Image("Histogram with differences between sensor %s and sensor %s applying Yeo-Johnson power transformation" % (sensorId, diffId), plot_url)
        all_imgs.append(img)
        """
        
        file_name = "reportPvalueS%s" % sensorId
        Stats.init_report(file_name, all_imgs)
        
        
        return pvalue, crit
        
    def test_all(alfa = 0.05):
         sensors_ids = Stats.getFlowSensorsIds()
         sensors = list()
         start_time = time.time()
         for sensor_id in sensors_ids:
             neighbors = Stats.getSensorNeighbors()[sensor_id]
             sensor_type = "flow"
             network_type = "tg"
             pvalues, pvalues_neighbors = Stats.test(sensor_id)
             sensor = Sensor(sensor_id, neighbors,sensor_type, network_type)
             sensors.append(sensor)
         elapsed_time = time.time() - start_time
         print("elapsed_time" + str(elapsed_time) + "seconds")
             
    def create_temporary_table(cursor):
        query = """
        CREATE TEMPORARY TABLE `differencesmeasuretg` (
  `iddifferencesmeasure` int(11) NOT NULL AUTO_INCREMENT,
  `idSensor` int(11) NOT NULL,
  `difference` float DEFAULT NULL,
  `instant` datetime NOT NULL,
  `diffWith` int(11) NOT NULL,
  PRIMARY KEY (`iddifferencesmeasure`),
  KEY `idx_instant` (`instant`)
) ENGINE=InnoDB AUTO_INCREMENT=6214810 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        """
        cursor.execute(query)
        
    
    def test(alfa = 0.05, sensorId=None):

        db, cursor = Stats.make_connection("infraquinta")
        network = Network("infraquinta")
        network.addAllEvents(db, cursor)
     
        print("Number of events:", network.countEvent())
        begin = datetime.datetime(2017, 5, 1)
        end = datetime.datetime(2017, 6, 16)
       
        #Stats.create_temporary_table(cursor)
        start_time = time.time()
        #Stats.write_to_disk_differences(db, cursor, sensorId)
        elapsed_time = time.time() - start_time
        critsList = []
        pvaluesList = []
        pvaluesDict = dict()
       
       
        ids_compare = Stats.getSensorNeighbors()[sensorId] 
        
        query = "SELECT difference FROM differencesmeasuretg WHERE idSensor=%s and diffWith=%s"
   
        cursor.execute(query, (sensorId, sensorId))
      
        measures = list()
    
        for element in cursor:
            measures.append(element[0])
           
        pvalue, crit = Stats.pvalue_analysis(measures, sensorId, sensorId, cursor, db)
        Stats.file.append(Text.Text("Pvalue is:" + str(pvalue)))
        Stats.file.append(Text.Text("Critical value is:" + str(crit)))
     
       
        for diffId in ids_compare:
            query = "SELECT difference FROM differencesmeasuretg WHERE idSensor=%s and diffWith=%s"
            
            cursor.execute(query, (sensorId, diffId))
            
            
            measures = list(measure[0] for measure in cursor)
            pvalues, crits = Stats.pvalue_analysis(measures, sensorId, diffId, cursor, db)
         
            critsList.append(crits)
            pvaluesList.append(pvalues)
            pvaluesDict[diffId] = pvalues
        
        if pvalue < alfa:
            #there is an event
           
            #is there a spatial consensus? 
            if Pvalue.consensus(pvaluesList, critsList, alfa):
                print("common event")
            else:
                print("point failure")
                
            #identify measures that are anomalies
          
            TP = 0
            FP = 0
            
            start_time = time.time()
        
                
            query = """
                    SELECT anomaly, date 
                    FROM infraquinta.differencesmeasuretg as DFT, infraquinta.anomaliestg as ATG
                    WHERE DFT.idSensor=%s AND  DFT.diffWith=%s and (DFT.difference< %s OR DFT.difference>%s) and DFT.idSensor = ATG.idSensor
                    AND DFT.instant = ATG.date
                    """
           
            cursor.execute(query, (sensorId, sensorId, float(-crit), float(crit)))
            
            dates = [el[1] for el in cursor]
            print("all_dates", len(dates))
            detected_events = dict()
            for date in dates:
                    event = network.findEvent(date)
                    if event != None:
                        detected_events[event.getId()] = event
                        TP += 1
                        
                    else:
                        FP += 1
            
            print("No detected events", len(list(detected_events.keys())))
            encoded = network.plot_timeline_events(begin, end, dates)
            img = image.Image("Timeline of events for sensor %s" % (sensorId), encoded)
            Stats.file.append(img)
            
            query = """
                SELECT anomaly, date 
                FROM infraquinta.differencesmeasuretg as DFT, infraquinta.anomaliestg as ATG
                WHERE DFT.idSensor=%s AND  DFT.diffWith=%s and (DFT.difference > %s OR DFT.difference = %s OR DFT.difference = %s OR DFT.difference < %s) and DFT.idSensor = ATG.idSensor
                AND DFT.instant = ATG.date
                """
                    
            cursor.execute(query, (sensorId, sensorId, float(-crit), float(-crit), float(crit), float(crit)))
            
            TN = 0
            FN = 0
    
            anomalies = [el[0] for el in cursor]
       
            
            for anomaly in anomalies:
                if anomaly == 1:
                    FN += 1
                else:
                    TN += 1
                    
            elapsed_time = time.time() - start_time
            print("elapsed_time" + str(elapsed_time) + "seconds")
            text_TP = Text.Text("True positive" + str(TP))
            text_FP = Text.Text("False Positive" + str(FP))
            Stats.file.append(text_TP)
            Stats.file.append(text_FP)
            Stats.file.append(Text.Text("False Negative:" + str(FN)))
            Stats.file.append(Text.Text("True Negative:" + str(TN)))
            precision = TP / (TP + FP)
            Stats.file.append(Text.Text("Precision:" + str(precision)))
            print("True positive", TP)
            print("False positive", FP)
   
        file_name = "reportPvalueS%s" % sensorId
        path = "F:/Tese/exploratory/wisdom/reports_files/report_pvalue/%s.html" % file_name
        
        Stats.write_report(path)
        
        cursor.close()
        db.close()
       
        return  pvalue, pvaluesDict
        
            

       
        
        
#todo
      
        
    
   