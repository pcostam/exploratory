# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:41:53 2020

@author: anama
"""
import mysql.connector
from Event import Event
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import base64
from io import BytesIO
class Network:
  
    sensors = list()
    events = list()
 
    def __init__(self, name):
        self.name = name
 
    def getEvents(self):
        return self.events
    
    def getEndDate(self):
        return self.end_date_data
    
    def getBeginDate(self):
        return self.begin_date_data
    
    def addEvent(self, event):
        self.events.append(event)
        
    def addAllEvents(self, db, cursor):
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
                        
                except IndexError as error:
                    # Output expected IndexErrors.
                    pass

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
    
 
    def plot_timeline_events(self, begin_date, end_date, dates_events):
        print("PLOT")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.autofmt_xdate()
        print("begin_date", begin_date)
        print("end_date", end_date)
        
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
		     
                