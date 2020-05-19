# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:41:53 2020

@author: anama
"""
from base.Event import Event
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import base64
from io import BytesIO
from base import database
import pickle
from base.Sensor import Sensor
from preprocessing.series import preprocess_simulation
class Network:
    def __init__(self, name):
        self.name = name
        self.events = list()
        self.flowSensors = list()
        self.pressureSensors = list()
        self.sensors = list()

        if name == "infraquinta":
            self.flowSensorsIds = [1, 2, 4, 6, 9, 10, 12, 14]
            self.pressureSensorsIds = [3, 5, 7, 8, 11, 13, 15]
            self.network_type = "tg"
            #in order
            self.sensorNeighbors =  {1:[10,14,12], 2:[4,6,9,12], 3:[7,5,8,11], 4:[6,2,9,12], 5:[7,3,9,11], 
                                     6:[4,2,9,12], 7:[5,3,8,11], 8:[11,13,5,7], 9:[12, 14, 4, 6], 10:[1, 14, 12], 
                                     11:[13, 8, 15, 5, 7], 12:[14, 9, 4, 6], 13:[15, 11, 8], 14:[10, 12], 15:[13,11,8]}
         
            network_type = self.network_type
            for sensor_id in self.flowSensorsIds:
                neighbors = self.sensorNeighbors[sensor_id]
                sensor_type = "flow"
                sensor = Sensor(sensor_id, neighbors,sensor_type, network_type)  
                self.flowSensors.append(sensor)
                
            for sensor_id in self.pressureSensorsIds:
                neighbors = self.sensorNeighbors[sensor_id]
                sensor_type = "pressure"
                sensor = Sensor(sensor_id, neighbors,sensor_type, network_type)  
                self.pressureSensors.append(sensor)
                
            self.sensors += self.flowSensors
            self.sensors += self.pressureSensors
          
    def getSensorNeighbors(self):
        return self.sensorNeighbors
    
    def getSensorNeighborsById(self, idSensor):
        return self.getSensorNeighbors()[idSensor]
    

    def getSensorsIds(self):
        return self.flowSensorsIds + self.pressureSensorIds
    
    def getPressureSensorsIds(self):
        return self.pressureSensorsIds
    
    def getEvents(self):
        return self.events
    
    def getEndDate(self):
        return self.end_date_data
    
    def getBeginDate(self):
        return self.begin_date_data
    
    def addEvent(self, event):
        self.events.append(event)
    
    def addSimulatedEvents(self, df):
        isLeak = False
        for index, row in df.iterrows():
            if row['leak'] == 1:
                isLeak = True
                start = row['time']
            if isLeak and row['leak'] == 0:
                isLeak = False
                end = df['time'].iloc[index]
                self.addEvent(Event(start, end, "fuga"))
   
    def addAllEvents(self, db, cursor, dump=True):
        query = """
        SELECT descricao, date FROM ordemdata ORDER BY date ASC
        """
        cursor.execute(query)
        
        aux = list(cursor)
     
        self.end_date_data = aux[0][1]
        self.begin_date_data = aux[-1][1]
        
        size_aux = len(aux)
        for i in range(0,size_aux):
            descricao = aux[i][0]
            date = aux[i][1]
            if descricao == 'Percepcao':
                try:
                    #fuga
                    date = aux[i][1]
                    start = Network.date_N_days_ago(date, 2)
                    #end vai ser quando for abertura
                    j = Network.find_open(i, aux, "Abertura")
                    abertura = Event(aux[j][1], aux[j][1], aux[j][0], []) 
                    end = aux[j][1]
                    events = [abertura]
                    self.addEvent(Event(start, end, "fuga", events))
                    """
                    else:
                        #ocorreu apenas fecho. (reparacao?) nao sei se e' fuga
                        Event(date, date,"reparo", [])
                        Network.find_open(i, aux)
                    """
                        
                except IndexError:
                    # Output expected IndexErrors.
                    pass
        if dump:
            self.dumpEvents()
    
    def loadEvents(self):
        path = "F:/manual/Tese/exploratory/wisdom/base/network_events"
        with open (path, 'rb') as fp:
           self.events = pickle.load(fp)
        return self.events
        
    def dumpEvents(self):
        path = "F:/manual/Tese/exploratory/wisdom/base/network_events"
        with open(path, 'wb') as fp:
            pickle.dump(self.events, fp)
   
    def date_N_days_ago(date, days):
        return date - datetime.timedelta(days=days) 

    def find_open(init, aux, name_event):
         j = init
         while aux[j][0] != name_event:
             j += 1
         
         return j
    
    def countEvent(self):
        size = len(self.events)    
        return size
    
    def findEventType(self, type_event, date):
        size = len(self.events)
        for i in range(0, size):
            start = self.events[i].getStart()
            end = self.events[i].getEnd()
            if date >= start and date <= end and self.events[i].getName == type_event:
                return self.events[i]
        return None
     
    def findEvent(self, date):
        size = len(self.events)
        for i in range(0, size):
            start = self.events[i].getStart()
            end = self.events[i].getEnd()
            if date >= start and date <= end:
                return self.events[i]
        return None
    
    def select_events(self, start, end):
        start = datetime.strptime(start, '%d-%m-%Y %H:%M:%S')
        end = datetime.strptime(end, '%d-%m-%Y %H:%M:%S')
        size = len(self.events)
        sevents = list()
        for i in range(0, size):
            estart = self.events[i].getStart()
            eend = self.events[i].getEnd()
         
            if start <= estart and end >= eend:
                sevents.append(self.events[i])
        return sevents
                
        
    def plot_timeline_events(self, begin_date, end_date, dates_events):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.autofmt_xdate()
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        ax.get_yaxis().set_ticklabels(["Sensor"])
        ax.get_yaxis().set_ticks([0])
        x_ticks = list()
        for event in self.events:
            if event.getStart() >= begin_date and event.getEnd() <= end_date:
                print("new_event", event.getId())
                xstart = event.getStart()
                xend = event.getEnd()
                x_ticks.append(xstart)
                x_ticks.append(xend)
                print("start:", xstart)
                print("end:", xend)
                plt.axvspan(xstart, xend, facecolor='#2ca02c', alpha=0.5)
               
        ax.scatter(dates_events, [0]*len(dates_events), marker='s')
        day = pd.to_timedelta("1", unit='D')
        plt.xlim(begin_date - day, end_date + day)
        
        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_xticks(x_ticks)
       
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        
        plt.show()
        return encoded
    
def test():
    #db, cursor = database.make_connection("infraquinta")
    network = Network("infraquinta")
    #network.addAllEvents(db, cursor)
    events = network.loadEvents()
    
    new_df = preprocess_simulation('12', 18000)
    network.addSimulatedEvents(new_df)
    print("Number of events:", network.countEvent())
    
    start = "19-10-2017 04:45:00"  
    end = "30-12-2017 23:45:00"
    new_events = network.select_events(start, end)
    print("number new events", len(new_events))
   
    